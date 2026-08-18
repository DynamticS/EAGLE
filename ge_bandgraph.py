"""
BandGraphGE — Global Mixture of Graph Experts (frequency-band nodes).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_str(v, default=''):
    return str(v if v is not None else default).strip().lower()


class LinearProbeGE(nn.Module):
    """L0 sanity: flatten DE [B,C,F] → (BatchNorm ≈ StandardScaler) → Linear(C*F, num_classes).

    Sklearn LR ceiling uses StandardScaler; raw DE has huge band-scale gaps
    (delta ≫ gamma). Without BN, Adam+max-f1 early-stop collapses to majority
    before the linear ranking signal is learned.
    """

    def __init__(self, num_channels=32, num_bands=5, num_classes=2, use_bn=True):
        super().__init__()
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.in_dim = num_channels * num_bands
        self.bn = nn.BatchNorm1d(self.in_dim) if use_bn else nn.Identity()
        self.fc = nn.Linear(self.in_dim, num_classes)
        print(
            f"[CONFIG-ECHO] LinearProbeGE in_dim={self.in_dim} "
            f"num_classes={num_classes} use_bn={use_bn}"
        )

    def forward(self, x):
        if x.dim() == 4:
            if x.size(2) == self.num_bands and x.size(3) == self.num_channels:
                x = x.mean(dim=1)  # [B,F,C]
                x = x.reshape(x.size(0), -1)
            else:
                x = x.reshape(x.size(0), -1)
        elif x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        else:
            x = x.reshape(x.size(0), -1)
        return self.fc(self.bn(x))


class BandMHSA(nn.Module):
    """Multi-head self-attention over frequency-band nodes (axis F)."""

    def __init__(self, dim, heads=4, dim_head=16, dropout=0.1):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [N, F, C]
        N, Fnodes, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(N, Fnodes, h, self.dim_head).transpose(1, 2)
            for t in qkv
        ]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(N, Fnodes, h * self.dim_head)
        return self.to_out(out)


class GraphExpertMoE(nn.Module):
    """
    Shared Â; E experts with W_e ∈ R^{C×C_out}; node-level ST-hard / soft routing.
    Vectorized — no per-expert mask loops.
    """

    def __init__(self, in_dim, out_dim, num_experts, num_bands=5,
                 adj_nonneg='softplus', routing='st_hard',
                 self_loop='after_softplus', wg_init_std=0.02, diag_routing=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.num_bands = num_bands
        self.adj_nonneg = _norm_str(adj_nonneg, 'softplus')
        self.routing = _norm_str(routing, 'st_hard')
        self.self_loop = _norm_str(self_loop, 'after_softplus')
        self.diag_routing = bool(diag_routing)
        self.wg_init_std = float(wg_init_std)

        self.xs, self.ys = torch.tril_indices(num_bands, num_bands, offset=-1)
        A_free = torch.empty(num_bands, num_bands)
        nn.init.uniform_(A_free, 0.0, 1.0)
        self.A_free = nn.Parameter(A_free[self.xs, self.ys], requires_grad=True)

        # Per-expert kaiming: 3D tensor would use fan_in=in*out (wrong by ~√C)
        self.W = nn.Parameter(torch.empty(num_experts, in_dim, out_dim))
        for e in range(num_experts):
            nn.init.kaiming_uniform_(self.W[e], a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(num_experts, out_dim))

        self.W_g = nn.Parameter(torch.empty(in_dim, num_experts))
        nn.init.normal_(self.W_g, mean=0.0, std=self.wg_init_std)
        self.register_buffer('_W_g_init', self.W_g.detach().clone())
        # routing diag: counts[v, e]
        self.register_buffer('_route_counts', torch.zeros(num_bands, num_experts))
        self.register_buffer('_gamma_max_sum', torch.zeros(()))
        self.register_buffer('_gamma_n', torch.zeros(()))

    # Process-level accumulator across fold reinits (for heatmap dump)
    _accum_counts = None
    _accum_gamma_max_sum = 0.0
    _accum_gamma_n = 0.0
    _accum_wg_deltas = []

    def reset_routing_stats(self):
        self._route_counts.zero_()
        self._gamma_max_sum.zero_()
        self._gamma_n.zero_()

    @classmethod
    def reset_global_routing_accum(cls):
        cls._accum_counts = None
        cls._accum_gamma_max_sum = 0.0
        cls._accum_gamma_n = 0.0
        cls._accum_wg_deltas = []

    def flush_routing_to_global(self):
        c = self._route_counts.detach().cpu()
        if GraphExpertMoE._accum_counts is None:
            GraphExpertMoE._accum_counts = c.clone()
        else:
            GraphExpertMoE._accum_counts = GraphExpertMoE._accum_counts + c
        GraphExpertMoE._accum_gamma_max_sum += float(self._gamma_max_sum.item())
        GraphExpertMoE._accum_gamma_n += float(self._gamma_n.item())
        GraphExpertMoE._accum_wg_deltas.append(
            float((self.W_g.detach() - self._W_g_init).norm().item())
        )

    @classmethod
    def global_routing_report(cls, n_shuffle=100, seed=8989, num_bands=5, num_experts=5):
        """Report from process-level accumulated counts."""
        if cls._accum_counts is None:
            return None
        # temporarily wrap counts into a throwaway instance-like namespace
        counts = cls._accum_counts.double()
        total = counts.sum().clamp(min=1.0)
        p_ve = counts / total
        p_v = p_ve.sum(dim=1, keepdim=True)
        p_e = p_ve.sum(dim=0, keepdim=True)
        ratio = p_ve / (p_v * p_e).clamp(min=1e-12)
        mask = p_ve > 0
        mi = float((p_ve[mask] * torch.log(ratio[mask])).sum().item())

        flat_v, flat_e = [], []
        for v in range(counts.size(0)):
            for e in range(counts.size(1)):
                c = int(counts[v, e].item())
                flat_v.extend([v] * c)
                flat_e.extend([e] * c)
        flat_v = np.asarray(flat_v, dtype=np.int64)
        flat_e = np.asarray(flat_e, dtype=np.int64)
        rng = np.random.RandomState(seed)
        shuf = []
        if len(flat_e) > 0:
            for _ in range(n_shuffle):
                e2 = flat_e.copy()
                rng.shuffle(e2)
                cm = np.zeros_like(counts.numpy())
                for vv, ee in zip(flat_v, e2):
                    cm[vv, ee] += 1
                p = cm / max(cm.sum(), 1.0)
                pv = p.sum(1, keepdims=True)
                pe = p.sum(0, keepdims=True)
                r = p / np.clip(pv * pe, 1e-12, None)
                m = p > 0
                shuf.append(float((p[m] * np.log(r[m])).sum()))
        gmax = (cls._accum_gamma_max_sum / max(cls._accum_gamma_n, 1.0))
        return {
            'counts': counts.numpy(),
            'I_ve': mi,
            'I_shuffle_mean': float(np.mean(shuf)) if shuf else float('nan'),
            'I_shuffle_std': float(np.std(shuf)) if shuf else float('nan'),
            'gamma_max_mean': float(gmax),
            'expert_frac': (counts.sum(0) / total).tolist(),
            'W_g_delta_norm_mean': float(np.mean(cls._accum_wg_deltas)) if cls._accum_wg_deltas else float('nan'),
            'n_tokens': float(total.item()),
        }

    def build_A_hat(self, device, dtype):
        A = torch.zeros(self.num_bands, self.num_bands, device=device, dtype=dtype)
        A[self.xs, self.ys] = self.A_free
        A = A + A.T
        eye = torch.eye(self.num_bands, device=device, dtype=dtype)
        if self.self_loop == 'before_softplus':
            A = A + eye
            if self.adj_nonneg == 'abs':
                A = A.abs()
            else:
                A = F.softplus(A)
        else:
            # after_softplus (default): softplus then +I
            # Note: softplus(0)=0.693 on former zeros → diag becomes 1.693 after +I
            # (heavier self-loop than paper's unit I); keep for continuity.
            if self.adj_nonneg == 'abs':
                A = A.abs()
            else:
                A = F.softplus(A)
            A = A + eye
        d = A.sum(dim=1).clamp(min=1e-6)
        D_inv_sqrt = torch.diag(d.pow(-0.5))
        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, X):
        """X: [N, Fn, C] → Y: [N, Fn, C_out]"""
        E = self.num_experts
        A_hat = self.build_A_hat(X.device, X.dtype)

        gamma = torch.softmax(torch.einsum('nfc,ce->nfe', X, self.W_g), dim=-1)

        if self.routing == 'soft':
            route = gamma
            idx = gamma.argmax(dim=-1)
        else:
            idx = gamma.argmax(dim=-1)
            hard = F.one_hot(idx, E).to(dtype=gamma.dtype)
            route = hard + gamma - gamma.detach()  # straight-through

        if self.diag_routing:
            with torch.no_grad():
                # idx: [N, Fn] → counts [Fn, E]
                one = F.one_hot(idx, E).to(dtype=self._route_counts.dtype)
                self._route_counts += one.sum(dim=0)
                self._gamma_max_sum += gamma.max(dim=-1).values.sum()
                self._gamma_n += gamma.size(0) * gamma.size(1)

        support = torch.einsum('nfc,eco->nefo', X, self.W)
        support = support + self.bias.view(1, E, 1, -1)
        prop = torch.einsum('fg,nego->nefo', A_hat, support)
        Y = torch.einsum('nfe,nefo->nfo', route, prop)
        return Y

    def routing_report(self, n_shuffle=100, seed=8989):
        """Mutual information I(v;e) + shuffle baseline. Returns dict."""
        counts = self._route_counts.detach().cpu().double()
        total = counts.sum().clamp(min=1.0)
        p_ve = counts / total
        p_v = p_ve.sum(dim=1, keepdim=True)
        p_e = p_ve.sum(dim=0, keepdim=True)
        # I = Σ p log(p/(p_v p_e)); skip zeros
        ratio = p_ve / (p_v * p_e).clamp(min=1e-12)
        mask = p_ve > 0
        mi = float((p_ve[mask] * torch.log(ratio[mask])).sum().item())

        # shuffle baseline: permute expert assignments keeping node margins
        flat_v = []
        flat_e = []
        for v in range(self.num_bands):
            for e in range(self.num_experts):
                c = int(counts[v, e].item())
                flat_v.extend([v] * c)
                flat_e.extend([e] * c)
        flat_v = np.asarray(flat_v, dtype=np.int64)
        flat_e = np.asarray(flat_e, dtype=np.int64)
        rng = np.random.RandomState(seed)
        shuf = []
        if len(flat_e) > 0:
            for _ in range(n_shuffle):
                e2 = flat_e.copy()
                rng.shuffle(e2)
                cm = np.zeros((self.num_bands, self.num_experts), dtype=np.float64)
                for vv, ee in zip(flat_v, e2):
                    cm[vv, ee] += 1
                p = cm / cm.sum()
                pv = p.sum(1, keepdims=True)
                pe = p.sum(0, keepdims=True)
                r = p / np.clip(pv * pe, 1e-12, None)
                m = p > 0
                shuf.append(float((p[m] * np.log(r[m])).sum()))
        shuf_mean = float(np.mean(shuf)) if shuf else float('nan')
        shuf_std = float(np.std(shuf)) if shuf else float('nan')

        gamma_max_mean = float(
            (self._gamma_max_sum / self._gamma_n.clamp(min=1)).item()
        )
        wg_delta = float((self.W_g.detach() - self._W_g_init).norm().item())
        wg_init_n = float(self._W_g_init.norm().item())
        expert_frac = (counts.sum(0) / total).tolist()
        return {
            'counts': counts.numpy(),
            'I_ve': mi,
            'I_shuffle_mean': shuf_mean,
            'I_shuffle_std': shuf_std,
            'gamma_max_mean': gamma_max_mean,
            'expert_frac': expert_frac,
            'W_g_delta_norm': wg_delta,
            'W_g_init_norm': wg_init_n,
            'n_tokens': float(total.item()),
        }


class BandGraphBlock(nn.Module):
    """MHSA (first-order) → MoE-GCN (second-order) → node-Conv1d → LN+act+drop (+res)."""

    def __init__(self, dim, out_dim, num_experts, num_bands=5,
                 heads=4, dim_head=16, dropout=0.1,
                 adj_nonneg='softplus', routing='st_hard',
                 use_mhsa=True, use_node_conv=True, use_gcn=True,
                 self_loop='after_softplus', wg_init_std=0.02, diag_routing=False):
        super().__init__()
        self.use_mhsa = use_mhsa
        self.use_node_conv = use_node_conv
        self.use_gcn = use_gcn
        self.use_res = (dim == out_dim)

        self.mhsa = BandMHSA(dim, heads=heads, dim_head=dim_head, dropout=dropout) if use_mhsa else None

        if use_gcn:
            self.gcn = GraphExpertMoE(
                in_dim=dim, out_dim=out_dim, num_experts=num_experts,
                num_bands=num_bands, adj_nonneg=adj_nonneg, routing=routing,
                self_loop=self_loop, wg_init_std=wg_init_std, diag_routing=diag_routing,
            )
            self.proj = None
        else:
            self.gcn = None
            self.proj = nn.Identity() if dim == out_dim else nn.Linear(dim, out_dim)

        self.node_conv = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=0) if use_node_conv else None
        self.ln = nn.LayerNorm(out_dim)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, X):
        residual = X
        if self.mhsa is not None:
            X = X + self.mhsa(X)

        if self.gcn is not None:
            Y = self.gcn(X)
        else:
            Y = self.proj(X)

        if self.node_conv is not None:
            t = Y.transpose(1, 2)  # [N,C,Fn]
            t = F.pad(t, (1, 1), mode='replicate')
            Y = self.node_conv(t).transpose(1, 2)

        Y = self.drop(self.act(self.ln(Y)))
        if self.use_res:
            Y = Y + residual
        return Y


class TemporalPool(nn.Module):
    def __init__(self, dim, mode='mean'):
        super().__init__()
        self.mode = _norm_str(mode, 'mean')
        if self.mode == 'attn':
            self.query = nn.Parameter(torch.randn(dim))
        elif self.mode == 'conv':
            self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, T, C]
        if x.size(1) == 1 or self.mode == 'mean':
            return x.mean(dim=1)
        if self.mode == 'attn':
            scores = torch.einsum('btc,c->bt', x, self.query)
            w = torch.softmax(scores, dim=-1)
            return torch.einsum('bt,btc->bc', w, x)
        if self.mode == 'conv':
            t = self.conv(x.transpose(1, 2)).transpose(1, 2)
            return t.mean(dim=1)
        return x.mean(dim=1)


class BandGraphGE(nn.Module):
    """
    Paper §3.4 GE on frequency-band nodes.

    Input X: [B, T, F=5, C=32]
    Output: logits [B, num_classes]  (NO softmax)
    """

    def __init__(
        self,
        num_channels=32,
        num_bands=5,
        num_classes=2,
        num_windows=4,
        num_experts=5,
        num_layers=1,
        hidden=32,
        heads=4,
        dim_head=16,
        dropout=0.1,
        node_norm='ln',
        temporal='mean',
        adj_nonneg='softplus',
        routing='st_hard',
        delta_feat=False,
        stage='full',
        check_finite=False,
        use_mhsa='auto',
        use_node_conv='auto',
        self_loop='after_softplus',
        wg_init_std=0.02,
        diag_routing=False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.num_windows = int(num_windows)
        self.num_experts = int(num_experts)
        self.num_layers = int(num_layers)
        self.node_norm = _norm_str(node_norm, 'ln')
        self.temporal = _norm_str(temporal, 'mean')
        self.adj_nonneg = _norm_str(adj_nonneg, 'softplus')
        self.routing = _norm_str(routing, 'st_hard')
        self.self_loop = _norm_str(self_loop, 'after_softplus')
        self.wg_init_std = float(wg_init_std)
        self.diag_routing = bool(diag_routing)
        self.delta_feat = bool(delta_feat)
        self.stage = _norm_str(stage, 'full')
        self.check_finite = bool(check_finite)

        # Ladder: l1 / l2 / l3 / full
        if self.stage == 'l1':
            use_gcn, use_mhsa_b, use_node_conv_b = False, False, False
            n_exp = 1
        elif self.stage == 'l2':
            use_gcn, use_mhsa_b, use_node_conv_b = True, False, False
            n_exp = 1
        elif self.stage == 'l3':
            use_gcn, use_mhsa_b, use_node_conv_b = True, False, False
            n_exp = self.num_experts
        else:
            use_gcn, use_mhsa_b, use_node_conv_b = True, True, True
            n_exp = self.num_experts

        # Explicit L4 ablation overrides (auto = follow stage)
        mhsa_ov = _norm_str(use_mhsa, 'auto')
        conv_ov = _norm_str(use_node_conv, 'auto')
        if mhsa_ov == 'on':
            use_mhsa_b = True
        elif mhsa_ov == 'off':
            use_mhsa_b = False
        if conv_ov == 'on':
            use_node_conv_b = True
        elif conv_ov == 'off':
            use_node_conv_b = False

        use_mhsa, use_node_conv = use_mhsa_b, use_node_conv_b

        self._use_gcn = use_gcn
        self._n_exp = n_exp

        in_dim = num_channels * (2 if self.delta_feat else 1)
        self.in_dim = in_dim

        # L1 has no graph layers → keep channel dim through mean-pool
        if not (use_gcn or use_mhsa or use_node_conv):
            self.hidden = in_dim
            self.in_proj = nn.Identity()
            self.layers = nn.ModuleList()
            pool_dim = in_dim
        else:
            self.hidden = int(hidden)
            self.in_proj = nn.Identity() if in_dim == self.hidden else nn.Linear(in_dim, self.hidden)
            self.layers = nn.ModuleList([
                BandGraphBlock(
                    dim=self.hidden,
                    out_dim=self.hidden,
                    num_experts=n_exp,
                    num_bands=num_bands,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    adj_nonneg=self.adj_nonneg,
                    routing=self.routing,
                    use_mhsa=use_mhsa,
                    use_node_conv=use_node_conv,
                    use_gcn=use_gcn,
                    self_loop=self.self_loop,
                    wg_init_std=self.wg_init_std,
                    diag_routing=self.diag_routing,
                )
                for _ in range(max(self.num_layers, 1))
            ])
            pool_dim = self.hidden

        self.node_ln = nn.LayerNorm(in_dim) if self.node_norm == 'ln' else nn.Identity()
        self.band_embed = nn.Parameter(torch.zeros(num_bands, in_dim))
        nn.init.normal_(self.band_embed, std=0.02)
        self.window_embed = nn.Parameter(torch.zeros(max(self.num_windows, 1), in_dim))
        nn.init.normal_(self.window_embed, std=0.02)

        self.temporal_pool = TemporalPool(pool_dim, mode=self.temporal)
        self.fc = nn.Linear(pool_dim, num_classes)

        print(
            f"[CONFIG-ECHO] BandGraphGE stage={self.stage} "
            f"num_windows={self.num_windows} num_experts={n_exp} "
            f"num_layers={self.num_layers} hidden={self.hidden} "
            f"heads={heads} dim_head={dim_head} dropout={dropout} "
            f"node_norm={self.node_norm} temporal={self.temporal} "
            f"adj_nonneg={self.adj_nonneg} routing={self.routing} "
            f"self_loop={self.self_loop} wg_init_std={self.wg_init_std} "
            f"diag_routing={self.diag_routing} "
            f"delta_feat={self.delta_feat} use_gcn={use_gcn} "
            f"use_mhsa={use_mhsa} use_node_conv={use_node_conv} "
            f"in_dim={in_dim} pool_dim={pool_dim}"
        )

    def reset_routing_stats(self):
        for layer in self.layers:
            if getattr(layer, 'gcn', None) is not None:
                layer.gcn.reset_routing_stats()

    def routing_reports(self):
        """Collect routing_report() from each GraphExpertMoE layer."""
        out = []
        for i, layer in enumerate(self.layers):
            if getattr(layer, 'gcn', None) is not None and layer.gcn.diag_routing:
                r = layer.gcn.routing_report()
                r['layer'] = i
                out.append(r)
        return out

    def _maybe_delta(self, X):
        if not self.delta_feat:
            return X
        B, T, Fn, C = X.shape
        d = torch.zeros_like(X)
        if T > 1:
            d[:, 1:] = X[:, 1:] - X[:, :-1]
        return torch.cat([X, d], dim=-1)

    def forward(self, X):
        """
        X: [B, T, Fn, C] preferred.
        Also accepts [B, Fn, C] or legacy [B, C, Fn].
        """
        if X.dim() == 3:
            if X.size(1) == self.num_channels and X.size(2) == self.num_bands:
                X = X.transpose(1, 2).unsqueeze(1)
            else:
                X = X.unsqueeze(1)
        assert X.dim() == 4, f'expected [B,T,Fn,C], got {tuple(X.shape)}'
        B, T, Fn, _ = X.shape
        assert Fn == self.num_bands, f'Fn={Fn} != num_bands={self.num_bands}'

        X = self._maybe_delta(X)
        X = self.node_ln(X)
        X = X + self.band_embed.view(1, 1, Fn, -1)
        we = self.window_embed
        if T > we.size(0):
            we = torch.cat([we, we.new_zeros(T - we.size(0), we.size(1))], dim=0)
        X = X + we[:T].view(1, T, 1, -1)

        N = B * T
        H = X.reshape(N, Fn, -1)
        H = self.in_proj(H)

        for layer in self.layers:
            H = layer(H)
            if self.check_finite:
                assert torch.isfinite(H).all(), 'non-finite in BandGraphGE layer'

        # average pooling over band nodes (paper readout)
        H = H.mean(dim=1).view(B, T, -1)
        H = self.temporal_pool(H)
        logits = self.fc(H)
        if self.check_finite:
            assert torch.isfinite(logits).all(), 'non-finite logits'
        return logits


def build_ge_from_args(args, num_classes=2, num_channels=32, num_bands=5):
    """Factory used by MoE.__init__."""
    impl = _norm_str(getattr(args, 'ge_impl', 'bandgraph'), 'bandgraph')
    if impl == 'linear_probe':
        return LinearProbeGE(
            num_channels=num_channels, num_bands=num_bands, num_classes=num_classes,
        )
    if impl == 'bandgraph':
        nw = int(getattr(args, 'ge_num_windows', 4))
        assert nw in (1, 2, 4), (
            f'--ge-num-windows must be in {{1,2,4}}, got {nw}. '
            f'T≥8 forbidden: delta band (0.5–4Hz) has <1 period in 0.5s windows; '
            f'variance estimate is noise.'
        )
        return BandGraphGE(
            num_channels=num_channels,
            num_bands=num_bands,
            num_classes=num_classes,
            num_windows=nw,
            num_experts=int(getattr(args, 'ge_num_experts', 5)),
            num_layers=int(getattr(args, 'ge_num_layers', 1)),
            hidden=int(getattr(args, 'ge_hidden', 32)),
            heads=int(getattr(args, 'ge_heads', 4)),
            dim_head=int(getattr(args, 'ge_dim_head', 16)),
            dropout=float(getattr(args, 'ge_dropout', 0.1)),
            node_norm=_norm_str(getattr(args, 'ge_node_norm', 'ln'), 'ln'),
            temporal=_norm_str(getattr(args, 'ge_temporal', 'mean'), 'mean'),
            adj_nonneg=_norm_str(getattr(args, 'ge_adj_nonneg', 'softplus'), 'softplus'),
            routing=_norm_str(getattr(args, 'ge_routing', 'st_hard'), 'st_hard'),
            delta_feat=_norm_str(getattr(args, 'ge_delta_feat', 'off'), 'off') == 'on',
            stage=_norm_str(getattr(args, 'ge_stage', 'full'), 'full'),
            check_finite=_norm_str(getattr(args, 'ge_check_finite', 'off'), 'off') == 'on',
            use_mhsa=_norm_str(getattr(args, 'ge_use_mhsa', 'auto'), 'auto'),
            use_node_conv=_norm_str(getattr(args, 'ge_use_node_conv', 'auto'), 'auto'),
            self_loop=_norm_str(getattr(args, 'ge_self_loop', 'after_softplus'), 'after_softplus'),
            wg_init_std=float(getattr(args, 'ge_wg_init_std', 0.02) or 0.02),
            diag_routing=_norm_str(getattr(args, 'ge_diag_routing', 'off'), 'off') == 'on',
        )
    raise ValueError(f'unknown ge_impl={impl!r}')
