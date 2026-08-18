# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
# Author: David Rau
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import math
from eeg_filter import EEGBandFilterOptimized
from typing import NamedTuple, Optional


class EnAsResult(NamedTuple):
    """Explicit LE/GE split of DE tensors. GE must only consume de_for_ge / last_x_ge."""
    embedded: Optional[torch.Tensor]
    de_raw: Optional[torch.Tensor]
    de_for_ge: Optional[torch.Tensor]
    de_for_le: Optional[torch.Tensor]
    H_b: Optional[torch.Tensor] = None       # [B, P*C, H] original patch feats
    attn_out: Optional[torch.Tensor] = None  # [B, P*C, H]
    Z_b: Optional[torch.Tensor] = None       # [B, P, C*H] final EAPatch tokens
    de_patch: Optional[torch.Tensor] = None  # [B, P, C, 5] LE-only per-patch DE (None if trial)


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
   
        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
    
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
  
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0.0, expert_out='logits'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.soft = nn.Softmax(1)
        self.expert_out = expert_out  # 'logits' (paper) or 'softmax' (legacy A/B)

    def forward(self, x):
        # Empty expert assignment is valid: [0, D] passes through Linear safely
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if self.expert_out == 'softmax':
            # [legacy] double-softmax path kept for A/B only
            out = self.soft(out)
        return out


class LinearExpert(nn.Module):
    """Single linear layer expert (no hidden, no activation). LE ablation only."""
    def __init__(self, input_size, output_size, dropout=0.0, expert_out='logits'):
        super(LinearExpert, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.soft = nn.Softmax(1)
        self.expert_out = expert_out

    def forward(self, x):
        out = self.fc(self.dropout(x))
        if self.expert_out == 'softmax':
            out = self.soft(out)
        return out


class MoE(nn.Module):
    # Class-level: survives per-fold MoE re-init so [GE-scale] still fires
    # under small loaders (≈6–10 batches/epoch × few epochs < 100 per instance).
    _ge_fwd_global = 0

    def __init__(self, args, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4, 
                 num_node=32, patch_size=None, num_patch=None, feature_embed_dim=None):
        super(MoE, self).__init__()
        # P0: align with utils.get_model() which passes args first (B11)
        self.args = args
        self.branch = getattr(args, 'branch', 'le_ge')
        self.expert_out = getattr(args, 'expert_out', 'logits')
        self.expert_dropout = float(getattr(args, 'expert_dropout', 0.0) or 0.0)
        self.le_expert_arch = str(getattr(args, 'le_expert_arch', 'mlp')).lower()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.num_node = num_node
        self.num_patch = num_patch
        self.feature_embed_dim = feature_embed_dim
        
        self.ge_impl = str(getattr(args, 'ge_impl', 'bandgraph')).strip().lower()
        if self.ge_impl not in ('bandgraph', 'linear_probe'):
            raise ValueError(f'ge_impl must be bandgraph|linear_probe, got {self.ge_impl}')
        self.ge_num_windows = int(getattr(args, 'ge_num_windows', 4) or 4)
        assert self.ge_num_windows in (1, 2, 4), (
            f'--ge-num-windows must be in {{1,2,4}}, got {self.ge_num_windows}. '
            f'T≥8 forbidden: delta (0.5–4Hz) has <1 cycle in 0.5s windows.'
        )

        # LE-path-only knobs (must NOT affect de_raw / GE)
        self.le_token_proj_dim = int(getattr(args, 'le_token_proj', 0) or 0)
        self.le_token_source = str(getattr(args, 'le_token_source', 'z')).lower()
        self.le_token_norm = str(getattr(args, 'le_token_norm', 'none')).lower()
        self.le_probe_dump = str(getattr(args, 'le_probe_dump', 'off')).lower() == 'on'
        self._probe_buf = None  # filled when le_probe_dump; never alters compute
        self.split_half = 64  # for split_concat: each half → 64-d
        # Measurement head: linear/mlp bypass MoE to read EAPatch Z_b quality (LE path only)
        self.eapatch_head = str(getattr(args, 'eapatch_head', 'moe')).lower()
        if self.eapatch_head not in ('linear', 'mlp', 'moe'):
            raise ValueError(f'eapatch_head must be linear|mlp|moe, got {self.eapatch_head}')
        self.eapatch_simple_head = None

        self.EnAs_embedding = EnAsEmbedding(
            num_node=num_node, 
            patch_size=patch_size, 
            num_patch=num_patch, 
            hidden_dim=feature_embed_dim, 
            hidden_dim2=feature_embed_dim,
            diag_eapatch=(str(getattr(args, 'diag_eapatch', 'off')).lower() == 'on'),
            eapatch_residual=str(getattr(args, 'eapatch_residual', 'on')).lower(),
            eapatch_res_norm=str(getattr(args, 'eapatch_res_norm', 'ln')).lower(),
            eapatch_res_gamma=str(getattr(args, 'eapatch_res_gamma', 'learnable')).lower(),
            patch_act=str(getattr(args, 'patch_act', 'gelu')).lower(),
            patch_norm=str(getattr(args, 'patch_norm', 'ln')).lower(),
            ge_num_windows=self.ge_num_windows,
            de_norm=str(getattr(args, 'de_norm', 'none')).lower(),
            filter_proj_gain=float(getattr(args, 'filter_proj_gain', 0.1)),
            de_clamp=str(getattr(args, 'de_clamp', 'on')).lower(),
            eapatch_attn_norm=str(getattr(args, 'eapatch_attn_norm', 'none')).lower(),
            eapatch_res_gamma_init=float(getattr(args, 'eapatch_res_gamma_init', 1.0)),
            eapatch_de_concat=str(getattr(args, 'eapatch_de_concat', 'off')).lower(),
            eapatch_de_level=str(getattr(args, 'eapatch_de_level', 'trial')).lower(),
            eapatch_de_window=int(getattr(args, 'eapatch_de_window', 128)),
            eapatch_attn_dir=str(getattr(args, 'eapatch_attn_dir', 'e2h')).lower(),
            eapatch_fuse=str(getattr(args, 'eapatch_fuse', 'residual')).lower(),
            use_eeg_filter=bool(getattr(args, 'use_eeg_filter', True)),
            fs=int(getattr(args, 'fs', 128) or 128),
            # Wire CR heads; MHA dropout historically unwired (=0). Keep effective 0
            # so defaults match prior runs; args.dropout_rate is echoed only.
            num_heads_CR=int(getattr(args, 'num_heads_CR', 8) or 8),
            dropout_rate=0.0,
            dropout_rate_arg=float(getattr(args, 'dropout_rate', 0.0) or 0.0),
            patch_embed=str(getattr(args, 'patch_embed', 'linear')).lower(),
            patch_conv_kernel=int(getattr(args, 'patch_conv_kernel', 7) or 7),
            patch_conv_chans=int(getattr(args, 'patch_conv_chans', 32) or 32),
            patch_conv_layers=int(getattr(args, 'patch_conv_layers', 1) or 1),
            patch_conv_pool=str(getattr(args, 'patch_conv_pool', 'avg')).lower(),
        )
        # Token feat dim may grow with --eapatch-de-concat on (LE path only)
        self._enas_token_feat = int(getattr(self.EnAs_embedding, 'token_feat_dim', feature_embed_dim))
        self._de_patch_dim = num_node * 5  # C·5 = 160
        self._num_patch = num_patch
        self._num_node = num_node

        # GE / fusion controls (P6)
        # NOTE: do not .lower() layout token — 'tb_T'.lower()=='tb_t' broke the transpose branch
        self.ge_layout = str(getattr(args, 'ge_layout', 'legacy'))
        self.adj_nonneg = str(getattr(args, 'adj_nonneg', 'none')).lower()
        self.ge_pool = str(getattr(args, 'ge_pool', 'cls')).lower()
        print(
            f"[CONFIG-ECHO] MoE branch={self.branch} ge_impl={self.ge_impl} "
            f"ge_num_windows={self.ge_num_windows} "
            f"ge_layout={self.ge_layout!r} adj_nonneg={self.adj_nonneg} ge_pool={self.ge_pool} "
            f"ge_num_experts={getattr(args,'ge_num_experts',None)} "
            f"ge_num_layers={getattr(args,'ge_num_layers',None)} "
            f"ge_hidden={getattr(args,'ge_hidden',None)} "
            f"ge_heads={getattr(args,'ge_heads',None)} "
            f"ge_dim_head={getattr(args,'ge_dim_head',None)} "
            f"ge_dropout={getattr(args,'ge_dropout',None)} "
            f"ge_node_norm={getattr(args,'ge_node_norm',None)} "
            f"ge_temporal={getattr(args,'ge_temporal',None)} "
            f"ge_adj_nonneg={getattr(args,'ge_adj_nonneg',None)} "
            f"ge_routing={getattr(args,'ge_routing',None)} "
            f"ge_stage={getattr(args,'ge_stage',None)} "
            f"ge_delta_feat={getattr(args,'ge_delta_feat',None)} "
            f"le_routing={getattr(args,'le_routing',None)} "
            f"expert_dropout={self.expert_dropout} expert_out={self.expert_out} "
            f"eapatch_residual={getattr(args,'eapatch_residual',None)} "
            f"patch_act={getattr(args,'patch_act',None)} "
            f"fs_arg={getattr(args,'fs',None)} use_eeg_filter_arg={getattr(args,'use_eeg_filter',None)} "
            f"num_heads_CR_arg={getattr(args,'num_heads_CR',None)} "
            f"dropout_rate_arg={getattr(args,'dropout_rate',None)}"
        )

        if self.branch in ('le_ge', 'ge'):
            from ge_bandgraph import build_ge_from_args
            self.GMOE = build_ge_from_args(args, num_classes=output_size, num_channels=32, num_bands=5)
            if self.branch == 'le_ge' and self.ge_impl == 'bandgraph':
                _fs = str(getattr(args, 'ge_fuse_space', 'logit')).lower()
                if _fs == 'prob':
                    print(
                        "[WARN] --ge-fuse-space prob: BandGraphGE logits → softmax before fuse. "
                        "Prefer --ge-fuse-space logit."
                    )
        # soft_GMOE kept for --ge-fuse-space prob (legacy A/B); default logit skips it
        if self.branch == 'le_ge':
            self.soft_GMOE = nn.Softmax(1)
        self.fusion_alpha = float(getattr(args, 'fusion_alpha', 0.9))
        self.ge_fuse_space = str(getattr(args, 'ge_fuse_space', 'logit')).lower()
        self.fusion_scale = str(getattr(args, 'fusion_scale', 'temp')).lower()
        self.fusion_detach_scale = str(getattr(args, 'fusion_detach_scale', 'zscore')).lower()
        if self.fusion_detach_scale not in ('off', 'zscore', 'frozen'):
            raise ValueError(
                f'fusion_detach_scale must be off|zscore|frozen, got {self.fusion_detach_scale}'
            )
        if self.fusion_detach_scale == 'frozen':
            print('[WARN] --fusion-detach-scale frozen: use offline/frozen-branch protocol; '
                  'runtime path currently behaves like zscore on live logits')
        self.fusion_ge_ln = None
        self.fusion_ge_temp = None
        # F3: dynamic α (paper §3.5); default fixed keeps legacy behaviour
        self.fusion_alpha_mode = str(getattr(args, 'fusion_alpha_mode', 'fixed')).lower()
        if self.fusion_alpha_mode not in ('fixed', 'learn_scalar', 'learn_trial'):
            raise ValueError(
                f'fusion_alpha_mode must be fixed|learn_scalar|learn_trial, '
                f'got {self.fusion_alpha_mode}'
            )
        self.fusion_alpha_min = float(getattr(args, 'fusion_alpha_min', 0.5))
        self.fusion_alpha_init = float(getattr(args, 'fusion_alpha_init', 0.9))
        self.fusion_alpha_tau = float(getattr(args, 'fusion_alpha_tau', 1.0))
        self.fusion_alpha_warmup = int(getattr(args, 'fusion_alpha_warmup', 5))
        self.fusion_alpha_lr_scale = float(getattr(args, 'fusion_alpha_lr_scale', 0.1))
        self.fusion_alpha_prior = float(getattr(args, 'fusion_alpha_prior', 1e-3))
        self.fusion_alpha_ln = None
        self.fusion_alpha_linear = None
        self.fusion_alpha_logit = None  # learn_scalar free parameter (init 0)
        self._fusion_force_le = False
        self._branch_pretrain = False
        self.last_alpha = None
        self.last_alpha_stats = None
        self.last_alpha_prior_loss = None
        # Scheme A/C: branch deep-supervision / independent pretrain (default off)
        self.aux_branch_ce_loc = float(getattr(args, 'aux_branch_ce_loc', 0.0) or 0.0)
        self.aux_branch_ce_glob = float(getattr(args, 'aux_branch_ce_glob', 0.0) or 0.0)
        self.branch_pretrain_epochs = int(getattr(args, 'branch_pretrain_epochs', 0) or 0)
        self.last_y_loc_live = None
        self.last_y_glob_live = None
        if self.branch == 'le_ge':
            if self.fusion_scale == 'ln':
                self.fusion_ge_ln = nn.LayerNorm(output_size)
            elif self.fusion_scale == 'temp':
                # learnable scalar temperature on GE logits, init=1.0
                self.fusion_ge_temp = nn.Parameter(torch.ones(()))
            a_min = self.fusion_alpha_min
            a_init = self.fusion_alpha_init
            if not (0.0 <= a_min < 1.0):
                raise ValueError(f'fusion_alpha_min must be in [0,1), got {a_min}')
            if not (a_min < a_init <= 1.0):
                raise ValueError(
                    f'fusion_alpha_init must be in (alpha_min, 1], got init={a_init} min={a_min}'
                )
            # b0 = logit((α_init − α_min)/(1 − α_min)); step-0 α ≡ α_init when s=0
            frac = (a_init - a_min) / max(1.0 - a_min, 1e-12)
            frac = min(max(frac, 1e-6), 1.0 - 1e-6)
            b0 = math.log(frac / (1.0 - frac))
            self.register_buffer('fusion_alpha_b0', torch.tensor(float(b0)))
            d_z = int(num_node) * int(self._enas_token_feat)
            if self.fusion_alpha_mode == 'learn_trial':
                self.fusion_alpha_ln = nn.LayerNorm(d_z)
                self.fusion_alpha_linear = nn.Linear(d_z, 1)
                nn.init.zeros_(self.fusion_alpha_linear.weight)
                nn.init.zeros_(self.fusion_alpha_linear.bias)
            elif self.fusion_alpha_mode == 'learn_scalar':
                self.fusion_alpha_logit = nn.Parameter(torch.zeros(()))
            print(
                f'[MoE] fusion: mode={self.fusion_alpha_mode} alpha={self.fusion_alpha} '
                f'space={self.ge_fuse_space} scale={self.fusion_scale} '
                f'detach_scale={self.fusion_detach_scale} '
                f'α_min={a_min} α_init={a_init} τ={self.fusion_alpha_tau} '
                f'warmup={self.fusion_alpha_warmup} prior={self.fusion_alpha_prior} '
                f'lr_scale={self.fusion_alpha_lr_scale} b0={b0:.4f}'
            )
        self.last_y_loc = None
        self.last_y_glob = None
        self.last_fuse_logit_stats = None
        self._fuse_alpha_override = None  # test-time α* for val_select
        self.le_routing = str(getattr(args, 'le_routing', 'patch')).lower()
        # I4: patch positional embedding (LE path only; does not affect GE)
        self.le_pos_embed = str(getattr(args, 'le_pos_embed', 'none')).lower()
        self._le_pos_dim = 32
        self.le_pos_param = None
        self.le_pos_proj = None
        # Do NOT assign self.le_pos_sincos = None before register_buffer (KeyError).
        if self.le_pos_embed not in ('none', 'learn', 'sincos'):
            raise ValueError(f'le_pos_embed must be none|learn|sincos, got {self.le_pos_embed}')
        if self.branch != 'ge' and self.le_pos_embed != 'none':
            P = int(num_patch)
            d_z = int(num_node) * int(self._enas_token_feat)
            if self.le_pos_embed == 'learn':
                self.le_pos_param = nn.Parameter(torch.zeros(1, P, self._le_pos_dim))
                nn.init.normal_(self.le_pos_param, std=0.02)
                self.le_pos_proj = nn.Linear(self._le_pos_dim, d_z)
            else:
                pe = torch.zeros(P, d_z)
                position = torch.arange(P, dtype=torch.float32).unsqueeze(1)
                div = torch.exp(torch.arange(0, d_z, 2, dtype=torch.float32) * (-math.log(10000.0) / d_z))
                pe[:, 0::2] = torch.sin(position * div)
                pe[:, 1::2] = torch.cos(position * div[:pe[:, 1::2].shape[1]])
                self.register_buffer('le_pos_sincos', pe.unsqueeze(0))  # [1,P,D]
            print(f'[MoE] le_pos_embed={self.le_pos_embed} P={P} d_z={d_z}')
        else:
            # explicit absence for forward checks
            self.register_buffer('le_pos_sincos', torch.empty(0), persistent=False)


        # LE experts / gating — skip for branch=ge
        if self.branch == 'ge':
            self.token_dim = num_node * self._enas_token_feat
            self.embedded_dim = self.token_dim
            self.le_token_proj = None
            self.le_z_proj = None
            self.le_de_proj = None
            self.le_z_ln = None
            self.le_de_ln = None
            self.token_norm = None
            self.experts = nn.ModuleList()
            self.w_gate = None
            self.w_noise = None
            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(1)
            self.register_buffer("mean", torch.tensor([0.0]))
            self.register_buffer("std", torch.tensor([1.0]))
            self.last_expert_frac = None
            self.last_aux_loss = None
            self.last_ge_scale = None
            self.last_patch_diversity = None
            print(f'[MoE] branch=ge (LE skipped) ge_impl={self.ge_impl} '
                  f'ge_layout={self.ge_layout} ge_pool={self.ge_pool} adj_nonneg={self.adj_nonneg}')
            return

        # P1: patch-level token dim — may switch to de_patch / concat / split_concat (LE path only)
        z_dim = (
            num_patch * num_node * self._enas_token_feat
            if self.le_routing == 'trial'
            else num_node * self._enas_token_feat  # D = C·H, default 32*64=2048
        )
        de_dim = self._de_patch_dim  # 160
        self.le_z_proj = None
        self.le_de_proj = None
        self.le_z_ln = None
        self.le_de_ln = None
        if self.le_token_source == 'de_patch':
            self.le_token_proj = None  # unused; experts see 160-d DE
            self.token_dim = de_dim
            print(f'[MoE] le_token_source=de_patch token_dim={self.token_dim} '
                  f'(LE path only; does not affect GE)')
        elif self.le_token_source == 'concat':
            # [z via optional proj] ⊕ de_patch
            if self.le_token_proj_dim > 0:
                self.le_token_proj = nn.Linear(z_dim, self.le_token_proj_dim)
                z_part = self.le_token_proj_dim
                print(f'[MoE] le_token_proj: {z_dim} → {self.le_token_proj_dim} '
                      f'(LE path only; does not affect GE)')
            else:
                self.le_token_proj = None
                z_part = z_dim
            self.token_dim = z_part + de_dim
            print(f'[MoE] le_token_source=concat token_dim={self.token_dim} '
                  f'(z_part={z_part}+de={de_dim}; LE path only)')
        elif self.le_token_source == 'split_concat':
            # LN(Linear(z,64)) ⊕ LN(Linear(de_patch,64)) — equal scale/dim halves
            half = int(self.split_half)
            self.le_token_proj = None
            self.le_z_proj = nn.Linear(z_dim, half)
            self.le_de_proj = nn.Linear(de_dim, half)
            self.le_z_ln = nn.LayerNorm(half)
            self.le_de_ln = nn.LayerNorm(half)
            self.token_dim = half * 2
            print(f'[MoE] le_token_source=split_concat token_dim={self.token_dim} '
                  f'(LN(Linear(z,{half}))⊕LN(Linear(de,{half})); LE path only)')
        else:
            # z (default)
            if self.le_token_proj_dim > 0:
                self.le_token_proj = nn.Linear(z_dim, self.le_token_proj_dim)
                self.token_dim = self.le_token_proj_dim
                print(f'[MoE] le_token_proj: {z_dim} → {self.le_token_proj_dim} '
                      f'(LE path only; does not affect GE)')
            else:
                self.le_token_proj = None
                self.token_dim = z_dim
            print(f'[MoE] le_token_source=z token_dim={self.token_dim}')

        self.embedded_dim = self.token_dim  # keep attr name for any external refs
        # Optional norm on tokens before gate/experts (LE path only)
        # split_concat already LN's each half — skip outer norm unless explicitly requested
        if self.le_token_source == 'split_concat' and self.le_token_norm == 'none':
            self.token_norm = None
        elif self.le_token_norm == 'ln':
            self.token_norm = nn.LayerNorm(self.token_dim)
        elif self.le_token_norm == 'bn':
            self.token_norm = nn.BatchNorm1d(self.token_dim)
        else:
            self.token_norm = None
        if self.token_norm is not None:
            print(f'[MoE] le_token_norm={self.le_token_norm} (LE path only; does not affect GE)')

        # ---- eapatch-head: linear/mlp skip sparse MoE (measurement tool; GE untouched) ----
        if self.eapatch_head in ('linear', 'mlp'):
            # Head consumes mean_P(Z_b): dim = C·H (= z_dim under patch routing + source=z)
            d_head = int(num_node) * int(self._enas_token_feat)
            if self.eapatch_head == 'linear':
                self.eapatch_simple_head = nn.Linear(d_head, self.output_size)
            else:
                self.eapatch_simple_head = nn.Sequential(
                    nn.Linear(d_head, 64),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, self.output_size),
                )
            self.experts = nn.ModuleList()
            self.w_gate = None
            self.w_noise = None
            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(1)
            self.register_buffer("mean", torch.tensor([0.0]))
            self.register_buffer("std", torch.tensor([1.0]))
            self.last_expert_frac = None
            self.last_aux_loss = None
            self.last_ge_scale = None
            self.last_patch_diversity = None
            n_params = sum(p.numel() for p in self.eapatch_simple_head.parameters())
            print(f'[MoE] eapatch_head={self.eapatch_head} d_in={d_head} '
                  f'head_params={n_params} (MoE experts SKIPPED; LE path only; does not affect GE)')
            return

        # instantiate experts (LE path only; GE untouched) — eapatch_head=moe
        if self.le_expert_arch == 'linear':
            self.experts = nn.ModuleList([
                LinearExpert(self.token_dim, self.output_size,
                             dropout=self.expert_dropout, expert_out=self.expert_out)
                for _ in range(self.num_experts)
            ])
            print(f'[MoE] le_expert_arch=linear (no hidden/activation; LE path only)')
        else:
            self.experts = nn.ModuleList([
                MLP(self.token_dim, self.output_size, self.hidden_size,
                    dropout=self.expert_dropout, expert_out=self.expert_out)
                for _ in range(self.num_experts)
            ])
        self.w_gate = nn.Parameter(torch.empty(self.token_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.token_dim, num_experts), requires_grad=True)
        # B8: non-zero gate init — zeros collapse eval-time topk to fixed expert indices
        nn.init.normal_(self.w_gate, mean=0.0, std=0.02)
        # self.shape_embed = ShapeEmbedLayer(seq_len=self.input_size)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.last_expert_frac = None  # monitoring: length-E fraction of tokens
        self.last_aux_loss = None
        self.last_ge_scale = None  # dict: LE/GE magnitude + contrib ratio
        self.last_patch_diversity = None  # mean #distinct experts used per trial (patch mode)
        assert(self.k <= self.num_experts)
        print(f'[MoE] eapatch_head=moe le_routing={self.le_routing} token_dim={self.token_dim} '
              f'num_experts={self.num_experts} k={self.k} le_expert_arch={self.le_expert_arch}')

    def cv_squared(self, x):

        eps = 1e-9
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
   
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
      
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        # if torch.isnan(clean_values).any():
        #     print("Warning: clean_values contains NaN values")
        # if torch.isnan(noisy_values).any():
        #     print("Warning: noisy_values contains NaN values")
        # if torch.isnan(noise_stddev).any():
        #     print("Warning: noise_stddev contains NaN values")
        # if torch.isnan(noisy_top_values).any():
        #     print("Warning: noisy_top_values contains NaN values")
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-9):
   
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # P1/B6: top-k on RAW logits; softmax only within top-k (equiv. to global-softmax then re-norm)
        # [legacy buggy] logits = self.softmax(logits) then topk mixed scales into _prob_in_top_k
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def _run_ge(self, de_features):
        """Dispatch GE forward by ge_impl. Returns logits (no softmax)."""
        if self.ge_impl == 'linear_probe':
            ge_in = de_features  # [B,C,5] → flatten inside LinearProbeGE
        elif self.ge_impl == 'bandgraph':
            x_ge = getattr(self.EnAs_embedding, 'last_x_ge', None)
            if x_ge is None:
                ge_in = de_features.transpose(1, 2).unsqueeze(1).contiguous()
            else:
                ge_in = x_ge
        else:
            raise ValueError(f'unknown ge_impl={self.ge_impl!r}')
        if not hasattr(self, '_ge_in_printed'):
            print(f"[GE-in] ge_impl={self.ge_impl} shape={tuple(ge_in.shape)} "
                  f"layout={self.ge_layout!r} de_features={tuple(de_features.shape)}")
            self._ge_in_printed = True
        return self.GMOE(ge_in)

    def _maybe_store_probe(self, result: EnAsResult):
        """Side-channel dump for --le-probe-dump; must not alter forward compute."""
        if not self.le_probe_dump:
            return
        with torch.no_grad():
            self._probe_buf = {
                'de_raw': None if result.de_raw is None else result.de_raw.detach().cpu(),
                'H_b': None if result.H_b is None else result.H_b.detach().cpu(),
                'attn_out': None if result.attn_out is None else result.attn_out.detach().cpu(),
                'Z_b': None if result.Z_b is None else result.Z_b.detach().cpu(),
                'de_patch': None if result.de_patch is None else result.de_patch.detach().cpu(),
            }

    def set_fusion_epoch(self, epoch: int):
        """Warmup: force α=1.0 (pure LE) for the first N epochs (train+eval).
        Also toggles branch-pretrain (independent CE, no fuse) when configured.
        """
        warm = int(getattr(self, 'fusion_alpha_warmup', 0) or 0)
        self._fusion_force_le = bool(
            self.branch == 'le_ge'
            and self.fusion_alpha_mode in ('learn_scalar', 'learn_trial')
            and warm > 0
            and int(epoch) <= warm
        )
        pre_n = int(getattr(self, 'branch_pretrain_epochs', 0) or 0)
        self._branch_pretrain = bool(
            self.branch == 'le_ge' and pre_n > 0 and int(epoch) <= pre_n
        )

    def _need_branch_live(self):
        """True when live y_loc/y_glob must stay on the graph (aux CE or pretrain)."""
        return bool(
            getattr(self, '_branch_pretrain', False)
            or float(getattr(self, 'aux_branch_ce_loc', 0.0) or 0.0) > 0.0
            or float(getattr(self, 'aux_branch_ce_glob', 0.0) or 0.0) > 0.0
        )

    def _store_branch_logits(self, y_loc, y_glob):
        """Detach for dumps; optionally keep live refs for aux/pretrain CE."""
        self.last_y_loc = y_loc.detach()
        self.last_y_glob = y_glob.detach()
        if self._need_branch_live():
            self.last_y_loc_live = y_loc
            self.last_y_glob_live = y_glob
        else:
            self.last_y_loc_live = None
            self.last_y_glob_live = None

    def fusion_alpha_parameters(self):
        """Params that should get lr × fusion_alpha_lr_scale."""
        out = []
        if self.fusion_alpha_linear is not None:
            out.extend(list(self.fusion_alpha_linear.parameters()))
        if self.fusion_alpha_ln is not None:
            out.extend(list(self.fusion_alpha_ln.parameters()))
        if self.fusion_alpha_logit is not None:
            out.append(self.fusion_alpha_logit)
        return out

    def _resolve_fusion_alpha(self, z):
        """Return α as float or [B] tensor in [α_min, 1]; record stats / prior."""
        B = z.size(0)
        device, dtype = z.device, z.dtype
        self.last_alpha_prior_loss = torch.zeros((), device=device, dtype=dtype)

        if self._fuse_alpha_override is not None:
            a = float(self._fuse_alpha_override)
            alpha = torch.full((B,), a, device=device, dtype=dtype)
            self.last_alpha = alpha.detach()
            self.last_alpha_stats = {
                'mean': a, 'std': 0.0, 'min': a, 'max': a, 'forced': 'override',
            }
            return alpha

        if self._fusion_force_le:
            alpha = torch.ones(B, device=device, dtype=dtype)
            self.last_alpha = alpha.detach()
            self.last_alpha_stats = {
                'mean': 1.0, 'std': 0.0, 'min': 1.0, 'max': 1.0, 'forced': 'warmup',
            }
            return alpha

        a_min = float(self.fusion_alpha_min)
        a_init = float(self.fusion_alpha_init)
        tau = max(float(self.fusion_alpha_tau), 1e-6)
        b0 = float(self.fusion_alpha_b0.item()) if hasattr(self, 'fusion_alpha_b0') else 0.0

        if self.fusion_alpha_mode == 'fixed' or self.fusion_alpha_mode not in (
            'learn_scalar', 'learn_trial',
        ):
            a = float(self.fusion_alpha)
            alpha = torch.full((B,), a, device=device, dtype=dtype)
        elif self.fusion_alpha_mode == 'learn_scalar':
            s = self.fusion_alpha_logit
            alpha = a_min + (1.0 - a_min) * torch.sigmoid(s / tau + b0)
            alpha = alpha.expand(B)
        else:  # learn_trial
            pooled = z.mean(dim=1)  # [B, D]
            pooled = self.fusion_alpha_ln(pooled)
            s = self.fusion_alpha_linear(pooled).squeeze(-1)
            alpha = a_min + (1.0 - a_min) * torch.sigmoid(s / tau + b0)

        if self.fusion_alpha_prior > 0 and self.fusion_alpha_mode in (
            'learn_scalar', 'learn_trial',
        ):
            self.last_alpha_prior_loss = self.fusion_alpha_prior * ((alpha - a_init) ** 2).mean()

        with torch.no_grad():
            ad = alpha.detach()
            self.last_alpha = ad
            self.last_alpha_stats = {
                'mean': float(ad.mean().item()),
                'std': float(ad.std(unbiased=False).item()) if ad.numel() > 1 else 0.0,
                'min': float(ad.min().item()),
                'max': float(ad.max().item()),
                'forced': None,
            }
        return alpha

    def _detach_zscore(self, y):
        """Per-batch z-score with detached μ/σ — blocks amplitude-compensation grads."""
        mu = y.mean(dim=0, keepdim=True).detach()
        sd = y.std(dim=0, keepdim=True, unbiased=False).detach().clamp_min(1e-6)
        return (y - mu) / sd

    def _prepare_fuse_logits(self, y_loc, y_glob):
        """Optional detach-zscore on both branches before convex combine."""
        raw_le_std = float(y_loc.std().item())
        raw_ge_std = float(y_glob.std().item())
        if self.fusion_detach_scale in ('zscore', 'frozen'):
            y_loc_f = self._detach_zscore(y_loc)
            y_glob_f = self._detach_zscore(y_glob)
        else:
            y_loc_f, y_glob_f = y_loc, y_glob
        norm_le_std = float(y_loc_f.std().item())
        norm_ge_std = float(y_glob_f.std().item())
        norm_le_abs = float(y_loc_f.abs().mean().item())
        norm_ge_abs = float(y_glob_f.abs().mean().item())
        self.last_fuse_logit_stats = {
            'raw_le_std': raw_le_std,
            'raw_ge_std': raw_ge_std,
            'norm_le_std': norm_le_std,
            'norm_ge_std': norm_ge_std,
            'norm_le_abs': norm_le_abs,
            'norm_ge_abs': norm_ge_abs,
            'detach_scale': self.fusion_detach_scale,
        }
        # Inflate warn only when detach-scale is off (raw amp can change effective α).
        # Under zscore, monitor post-norm std / contrib instead (see _log_ge_scale).
        if self.fusion_detach_scale == 'off' and raw_le_std > 10.0:
            self.last_fuse_logit_stats['warn_le_inflate'] = True
        return y_loc_f, y_glob_f

    def _fuse_branches(self, y_loc, y_glob, alpha):
        """y = α·y_loc + (1−α)·y_glob with α scalar or [B]; optional detach-zscore."""
        y_loc, y_glob = self._prepare_fuse_logits(y_loc, y_glob)
        if not torch.is_tensor(alpha):
            return y_loc * float(alpha) + y_glob * (1.0 - float(alpha))
        if alpha.dim() == 0:
            a = alpha
            return y_loc * a + y_glob * (1.0 - a)
        a = alpha.view(-1, *([1] * (y_loc.dim() - 1)))
        return y_loc * a + y_glob * (1.0 - a)

    def _log_ge_scale(self, y_loc, y_glob, alpha):
        with torch.no_grad():
            MoE._ge_fwd_global += 1
            n = MoE._ge_fwd_global
            le_std = float(y_loc.std().item())
            le_abs = float(y_loc.abs().mean().item())
            ge_std = float(y_glob.std().item())
            ge_abs = float(y_glob.abs().mean().item())
            if torch.is_tensor(alpha):
                a_mean = float(alpha.detach().float().mean().item())
            else:
                a_mean = float(alpha)
            st = getattr(self, 'last_fuse_logit_stats', None) or {}
            # Effective contrib: after zscore use post-norm |mean|; else raw.
            if self.fusion_detach_scale in ('zscore', 'frozen'):
                le_m = float(st.get('norm_le_abs', le_abs) or le_abs)
                ge_m = float(st.get('norm_ge_abs', ge_abs) or ge_abs)
            else:
                le_m, ge_m = le_abs, ge_abs
            le_contrib = a_mean * le_m
            ge_contrib = (1.0 - a_mean) * ge_m
            ratio = le_contrib / (ge_contrib + 1e-12)
            self.last_ge_scale = {
                'le_std': le_std, 'le_abs': le_abs,
                'ge_std': ge_std, 'ge_abs': ge_abs,
                'le_contrib': le_contrib, 'ge_contrib': ge_contrib,
                'ratio': ratio, 'alpha': a_mean,
                'space': self.ge_fuse_space, 'scale': self.fusion_scale,
                'detach_scale': self.fusion_detach_scale,
                'norm_le_std': st.get('norm_le_std'),
                'norm_ge_std': st.get('norm_ge_std'),
            }
            if n == 1 or n % 100 == 0:
                t_s = ''
                if self.fusion_ge_temp is not None:
                    t_s = f' temp={float(self.fusion_ge_temp.detach()):.3f}'
                n_s = ''
                if st:
                    n_s = (f' norm_std LE={st.get("norm_le_std", float("nan")):.4f} '
                           f'GE={st.get("norm_ge_std", float("nan")):.4f}')
                warn = ''
                if self.fusion_detach_scale == 'off':
                    if le_std > 10.0:
                        warn = ' WARN_LE_INFLATE'
                else:
                    # Post-zscore std should be ~1; flag residual scale issues / contrib skew
                    n_le = float(st.get('norm_le_std', 1.0) or 1.0)
                    n_ge = float(st.get('norm_ge_std', 1.0) or 1.0)
                    if n_le > 2.5 or n_ge > 2.5:
                        warn = ' WARN_FUSE_NORM_STD'
                    elif 0.02 < a_mean < 0.98 and (ratio > 20.0 or ratio < 0.05):
                        # Skip extremes (warmup α=1 / degenerate α≈0) — expected zero contrib
                        warn = ' WARN_FUSE_CONTRIB'
                print(
                    f'[GE-scale] batch={n} space={self.ge_fuse_space} '
                    f'scale={self.fusion_scale} detach={self.fusion_detach_scale}{t_s} '
                    f'y_loc std={le_std:.4f} |mean|={le_abs:.4f} '
                    f'y_glob std={ge_std:.4f} |mean|={ge_abs:.4f} '
                    f'eff_contrib {a_mean:.2f}*|LE|={le_contrib:.4f} vs '
                    f'{1 - a_mean:.2f}*|GE|={ge_contrib:.4f} '
                    f'ratio(LE:GE)={ratio:.1f}x{n_s}{warn}'
                )

    def _build_le_tokens(self, z, result: EnAsResult):
        """Build routing tokens from Z_b and/or per-patch DE (LE path only)."""
        B, P, D = z.shape
        de_flat = None
        if result.de_patch is not None:
            # [B,P,C,5] → [B,P,160]
            de_flat = result.de_patch.reshape(B, P, -1)
        elif self.le_token_source in ('de_patch', 'concat', 'split_concat'):
            raise RuntimeError(
                f'le_token_source={self.le_token_source} requires --eapatch-de-level patch '
                f'(de_patch is None)'
            )

        if self.le_routing == 'trial':
            z_tok = z.reshape(B, P * D)
            n_tok_per_trial = 1
            if de_flat is not None:
                de_tok = de_flat.reshape(B, P * de_flat.size(-1))
            else:
                de_tok = None
        else:
            z_tok = z.reshape(B * P, D)
            n_tok_per_trial = P
            if de_flat is not None:
                de_tok = de_flat.reshape(B * P, de_flat.size(-1))
            else:
                de_tok = None

        if self.le_token_source == 'de_patch':
            tok = de_tok
        elif self.le_token_source == 'concat':
            z_part = self.le_token_proj(z_tok) if self.le_token_proj is not None else z_tok
            tok = torch.cat([z_part, de_tok], dim=-1)
        elif self.le_token_source == 'split_concat':
            z_part = self.le_z_ln(self.le_z_proj(z_tok))
            de_part = self.le_de_ln(self.le_de_proj(de_tok))
            tok = torch.cat([z_part, de_part], dim=-1)
        else:
            tok = self.le_token_proj(z_tok) if self.le_token_proj is not None else z_tok

        if self.token_norm is not None:
            if isinstance(self.token_norm, nn.BatchNorm1d):
                tok = self.token_norm(tok)
            else:
                tok = self.token_norm(tok)
        return tok, n_tok_per_trial

    def forward(self, x, loss_coef=1e-2):
        # ---- GE-only: skip LE ----
        if self.branch == 'ge':
            result = self.EnAs_embedding(x, de_only=True)
            # GE MUST consume de_for_ge (= de_raw alias), never de_for_le
            y = self._run_ge(result.de_for_ge)  # logits (no soft_GMOE)
            self._maybe_store_probe(result)
            aux = torch.zeros((), device=y.device, dtype=y.dtype)
            self.last_aux_loss = 0.0
            return y, aux

        result = self.EnAs_embedding(x)
        x_embedded = result.embedded
        de_for_ge = result.de_for_ge  # GE path only
        self._maybe_store_probe(result)
        if self.branch == 'le_ge':
            X_gmoe = self._run_ge(de_for_ge)  # raw GE logits
            # B18: default fuse in logit space (paper §3.5). soft_GMOE kept for legacy A/B.
            # X_gmoe_normalized = self.soft_GMOE(X_gmoe)  # [legacy prob fuse]
            if self.ge_fuse_space == 'prob':
                y_glob = self.soft_GMOE(X_gmoe)
            else:
                y_glob = X_gmoe
            if self.fusion_ge_ln is not None:
                y_glob = self.fusion_ge_ln(y_glob)
            elif self.fusion_ge_temp is not None:
                y_glob = y_glob * self.fusion_ge_temp

        # P1/P5: patch-level routing — paper (b,p) tokens; trial = flat P·D
        if x_embedded.dim() == 2:
            z = x_embedded.unsqueeze(1)
        else:
            z = x_embedded  # [B, P, D]
        B, P, D = z.shape
        # I4: add patch positional embedding (LE only)
        if self.le_pos_embed == 'learn' and self.le_pos_param is not None:
            z = z + self.le_pos_proj(self.le_pos_param[:, :P, :])
        elif self.le_pos_embed == 'sincos' and getattr(self, 'le_pos_sincos', None) is not None\
                and self.le_pos_sincos.numel() > 0:
            z = z + self.le_pos_sincos[:, :P, :D]

        # ---- linear / mlp head: mean_P(Z_b) → classifier (no MoE) ----
        if self.eapatch_head in ('linear', 'mlp') and self.eapatch_simple_head is not None:
            z_pool = z.mean(dim=1)  # [B, C·H]
            y_loc = self.eapatch_simple_head(z_pool)
            aux_loss = torch.zeros((), device=y_loc.device, dtype=y_loc.dtype)
            self.last_aux_loss = 0.0
            self.last_expert_frac = None
            self.last_patch_diversity = None
            y = y_loc
            if self.branch == 'le_ge':
                self._store_branch_logits(y_loc, y_glob)
                if getattr(self, '_branch_pretrain', False):
                    # Independent pretrain: no fuse; train loop uses live branch CEs
                    return y_loc, aux_loss
                alpha = self._resolve_fusion_alpha(z)
                y = self._fuse_branches(y_loc, y_glob, alpha)
                self._log_ge_scale(y_loc, y_glob, alpha)
            return y, aux_loss

        z_tok, n_tok_per_trial = self._build_le_tokens(z, result)

        gates, load = self.noisy_top_k_gating(z_tok, self.training)  # [N, E]
        
        # calculate importance loss (over all tokens)
        importance = gates.sum(0)
        aux_loss = self.cv_squared(importance) + self.cv_squared(load)
        # P0/B7: do NOT multiply loss_coef here
        # loss *= loss_coef

        # monitoring: expert frac + per-trial expert-set diversity (patch mode)
        with torch.no_grad():
            frac = (gates > 0).float().mean(0)  # fraction of tokens routed to each expert
            self.last_expert_frac = frac.detach().cpu()
            self.last_aux_loss = float(aux_loss.detach().cpu())
            if self.le_routing == 'patch' and n_tok_per_trial > 1:
                # gates: [B*P, E] → for each trial, how many distinct experts used across P patches
                used = (gates > 0).view(B, P, self.num_experts)  # [B, P, E]
                n_exp_per_trial = used.any(dim=1).float().sum(dim=1)  # [B]
                self.last_patch_diversity = float(n_exp_per_trial.mean().item())
            else:
                # trial routing: one token → diversity = #experts selected by top-k (≤k)
                self.last_patch_diversity = float((gates > 0).float().sum(dim=1).mean().item())

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(z_tok)
        _ = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        # Handle all-empty combine edge case
        if all(eo.numel() == 0 for eo in expert_outputs):
            y_tok = torch.zeros(z_tok.size(0), self.output_size, device=z_tok.device, dtype=z_tok.dtype)
        else:
            y_tok = dispatcher.combine(expert_outputs)  # [N, K]
        if self.le_routing == 'trial':
            y_loc = y_tok  # [B, K]
        else:
            y_loc = y_tok.view(B, P, -1).mean(dim=1)  # paper: (1/P) Σ_p

        y = y_loc
        if self.branch == 'le_ge':
            self._store_branch_logits(y_loc, y_glob)
            if getattr(self, '_branch_pretrain', False):
                # Independent pretrain: no fuse; train loop uses live branch CEs
                return y_loc, aux_loss
            alpha = self._resolve_fusion_alpha(z)
            y = self._fuse_branches(y_loc, y_glob, alpha)
            self._log_ge_scale(y_loc, y_glob, alpha)
        return y, aux_loss


class DiagEapatchDone(Exception):
    """Raised after enough read-only EAPatch patch-variance prints; not an error."""
    pass


class EnAsEmbedding(nn.Module):
    # Shared counters so train/eval prints accumulate across fold reinits within one process
    _diag_train_n = 0
    _diag_eval_n = 0
    _DIAG_MAX_PER_MODE = 2
    _mag_print_n = 0
    _MAG_PRINT_MAX = 2
    _ge_input_echo_n = 0

    def __init__(self, num_node=32, patch_size=32, num_patch=16, hidden_dim=64, hidden_dim2=64,
                 use_eeg_filter=True, fs=128, diag_eapatch=False,
                 eapatch_residual='on', eapatch_res_norm='ln', eapatch_res_gamma='learnable',
                 patch_act='gelu', patch_norm='ln', ge_num_windows=1,
                 # ---- LE path only (must NOT touch de_raw / de_for_ge) ----
                 de_norm='none', filter_proj_gain=0.1, de_clamp='on',
                 eapatch_attn_norm='none', eapatch_res_gamma_init=1.0,
                 eapatch_de_concat='off',
                 eapatch_de_level='trial', eapatch_de_window=128,
                 eapatch_attn_dir='e2h', eapatch_fuse='residual',
                 num_heads_CR=8, dropout_rate=0.0, dropout_rate_arg=None,
                 # I1: true temporal conv patch embed (LE path only)
                 patch_embed='linear', patch_conv_kernel=7, patch_conv_chans=32,
                 patch_conv_layers=1, patch_conv_pool='avg_max'):
        super(EnAsEmbedding, self).__init__()
        self.num_node = num_node
        self.patch_size = patch_size
        self.num_patch = num_patch
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.use_eeg_filter = use_eeg_filter
        self.fs = int(fs)
        self.num_heads_CR = int(num_heads_CR)
        self.attn_dropout = float(dropout_rate)  # effective MHA dropout (legacy default 0)
        self.dropout_rate_arg = (
            float(dropout_rate_arg) if dropout_rate_arg is not None else self.attn_dropout
        )
        self.patch_embed = str(patch_embed).lower()
        self.patch_conv_kernel = int(patch_conv_kernel)
        self.patch_conv_chans = int(patch_conv_chans)
        self.patch_conv_layers = int(patch_conv_layers)
        self.patch_conv_pool = str(patch_conv_pool).lower()
        if self.patch_embed not in ('linear', 'tconv'):
            raise ValueError(f'patch_embed must be linear|tconv, got {self.patch_embed}')
        if self.patch_conv_pool not in ('avg', 'max', 'avg_max', 'std'):
            raise ValueError(f'patch_conv_pool must be avg|max|avg_max|std, got {self.patch_conv_pool}')
        if self.patch_conv_layers not in (1, 2):
            raise ValueError(f'patch_conv_layers must be 1|2, got {self.patch_conv_layers}')
        self.diag_eapatch = bool(diag_eapatch)
        self.eapatch_residual = str(eapatch_residual).lower()
        self.eapatch_res_norm = str(eapatch_res_norm).lower()
        self.eapatch_res_gamma = str(eapatch_res_gamma).lower()
        self.patch_act = str(patch_act).lower()
        self.patch_norm = str(patch_norm).lower()
        self.ge_num_windows = int(ge_num_windows) if ge_num_windows else 1
        self.de_norm = str(de_norm).lower()
        self.filter_proj_gain = float(filter_proj_gain)
        self.de_clamp = str(de_clamp).lower()
        self.eapatch_attn_norm = str(eapatch_attn_norm).lower()
        self.eapatch_res_gamma_init = float(eapatch_res_gamma_init)
        self.eapatch_de_concat = str(eapatch_de_concat).lower()
        self.eapatch_de_level = str(eapatch_de_level).lower()
        self.eapatch_de_window = int(eapatch_de_window)
        self.eapatch_attn_dir = str(eapatch_attn_dir).lower()
        self.eapatch_fuse = str(eapatch_fuse).lower()
        if self.eapatch_attn_dir not in ('e2h', 'h2e'):
            raise ValueError(f'eapatch_attn_dir must be e2h|h2e, got {self.eapatch_attn_dir}')
        if self.eapatch_fuse not in ('residual', 'tri'):
            raise ValueError(f'eapatch_fuse must be residual|tri, got {self.eapatch_fuse}')
        # delta lowest freq 0.5Hz: require ≥0.5 period → window >= fs/(2*0.5)=fs
        min_win = int(self.fs / (2 * 0.5))  # = fs
        if self.eapatch_de_level == 'patch' and self.eapatch_de_window < min_win:
            raise ValueError(
                f'--eapatch-de-window={self.eapatch_de_window} < min {min_win} '
                f'(need ≥0.5 period of delta 0.5Hz at fs={self.fs}; '
                f'short patches alone cannot estimate delta DE)'
            )
        self.last_x_ge = None
        self.last_de_raw = None
        self.last_de_for_ge = None
        self.last_de_for_le = None
        self.last_de_patch = None  # [B,P,C,5] LE-only
        self.last_H_b = None
        self.last_attn_out = None
        self.last_Z_b = None
        self.last_fuse_stats = None
        print(
            f"[CONFIG-ECHO] EnAsEmbedding use_eeg_filter={self.use_eeg_filter} fs={self.fs} "
            f"ge_num_windows={self.ge_num_windows} "
            f"patch_act={self.patch_act} patch_norm={self.patch_norm} "
            f"eapatch_residual={self.eapatch_residual} "
            f"eapatch_de_level={self.eapatch_de_level} eapatch_de_window={self.eapatch_de_window} "
            f"eapatch_attn_dir={self.eapatch_attn_dir} eapatch_fuse={self.eapatch_fuse} "
            f"de_norm={self.de_norm} filter_proj_gain={self.filter_proj_gain} "
            f"de_clamp={self.de_clamp} eapatch_attn_norm={self.eapatch_attn_norm} "
            f"eapatch_res_gamma_init={self.eapatch_res_gamma_init} "
            f"eapatch_de_concat={self.eapatch_de_concat} "
            f"(LE-path-only; GE uses de_raw untouched) "
            f"num_heads_CR={self.num_heads_CR} attn_dropout={self.attn_dropout} "
            f"dropout_rate_arg={self.dropout_rate_arg} "
            f"patch_embed={self.patch_embed} k={self.patch_conv_kernel} "
            f"chans={self.patch_conv_chans} layers={self.patch_conv_layers} "
            f"pool={self.patch_conv_pool} "
            f"(attn_dropout kept 0 by default = prior unwired MHA behavior)"
        )
        if self.use_eeg_filter:
            original_signal_length = self.num_patch * self.patch_size
            self.eeg_filter = EEGBandFilterOptimized(
                num_channels=self.num_node,
                signal_length=original_signal_length,
                fs=self.fs
            )
            self.filter_feature_dim = 5
            self.filter_proj = nn.Linear(self.filter_feature_dim, self.hidden_dim // 2)
            torch.nn.init.xavier_uniform_(self.filter_proj.weight, gain=self.filter_proj_gain)
            if self.filter_proj.bias is not None:
                torch.nn.init.zeros_(self.filter_proj.bias)
            self.de_bn = nn.BatchNorm1d(self.filter_feature_dim) if self.de_norm == 'bn' else None
            self.de_ln = nn.LayerNorm(self.filter_feature_dim) if self.de_norm == 'ln' else None
        else:
            self.filter_feature_dim = 0
            self.de_bn = None
            self.de_ln = None

        # ---- Patch embed (I1): linear = legacy Conv1d(k=1) as Linear(32→H); tconv = true temporal ----
        if self.patch_embed == 'linear':
            self.nonlin_map = nn.Conv1d(self.patch_size, self.hidden_dim, kernel_size=1)
            self.patch_tconv = None
            self.patch_pool_avg = None
            self.patch_pool_max = None
            self.patch_tconv_proj = None
        else:
            self.nonlin_map = None
            k = self.patch_conv_kernel
            pad = k // 2
            ch = self.patch_conv_chans
            layers = []
            in_ch = 1
            for _ in range(self.patch_conv_layers):
                layers += [
                    nn.Conv1d(in_ch, ch, kernel_size=k, padding=pad),
                    nn.GELU(),
                ]
                in_ch = ch
            self.patch_tconv = nn.Sequential(*layers)
            self.patch_pool_avg = nn.AdaptiveAvgPool1d(1)
            self.patch_pool_max = nn.AdaptiveMaxPool1d(1)
            pool_dim = ch * (2 if self.patch_conv_pool == 'avg_max' else 1)
            self.patch_tconv_proj = nn.Linear(pool_dim, self.hidden_dim)
        self.nonlin_map2 = nn.Linear(self.hidden_dim, self.hidden_dim2)
        if self.patch_act == 'gelu':
            self.patch_activation = nn.GELU()
        elif self.patch_act == 'relu':
            self.patch_activation = nn.ReLU()
        else:
            self.patch_activation = None
        self.patch_ln = nn.LayerNorm(self.hidden_dim2) if self.patch_norm == 'ln' else None

        if self.use_eeg_filter:
            self.feature_fusion = nn.Linear(self.hidden_dim2 + self.hidden_dim // 2, self.hidden_dim2)
            # e2h (paper legacy): Q=Ê, K/V=H ; h2e: Q=H, K/V=Ê (output lives in entropy space)
            if self.eapatch_attn_dir == 'h2e':
                self.q_proj = nn.Linear(self.hidden_dim2, self.hidden_dim2)           # H → H
                self.k_proj = nn.Linear(self.hidden_dim // 2, self.hidden_dim2)       # H/2 → H
                self.v_proj = nn.Linear(self.hidden_dim // 2, self.hidden_dim2)       # H/2 → H
            else:
                self.q_proj = nn.Linear(self.hidden_dim // 2, self.hidden_dim2)       # H/2 → H
                self.k_proj = nn.Linear(self.hidden_dim2, self.hidden_dim2)           # H → H
                self.v_proj = nn.Linear(self.hidden_dim2, self.hidden_dim2)           # H → H
            if self.hidden_dim2 % self.num_heads_CR != 0:
                raise ValueError(
                    f'hidden_dim2={self.hidden_dim2} not divisible by num_heads_CR={self.num_heads_CR}'
                )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim2,
                num_heads=self.num_heads_CR,
                dropout=self.attn_dropout,
                batch_first=True,
            )
            g0 = float(self.eapatch_res_gamma_init)
            if self.eapatch_res_gamma == 'learnable':
                self.res_gamma = nn.Parameter(torch.tensor(g0))
            else:
                self.register_buffer('res_gamma', torch.tensor(g0))
            self.res_ln = nn.LayerNorm(self.hidden_dim2) if self.eapatch_res_norm == 'ln' else None
            self.attn_ln = nn.LayerNorm(self.hidden_dim2) if self.eapatch_attn_norm == 'ln' else None
            # tri-fuse: LN(H)+γ1·LN(attn)+γ2·LN(Ê·W_up)
            if self.eapatch_fuse == 'tri':
                self.fuse_h_ln = nn.LayerNorm(self.hidden_dim2)
                self.fuse_a_ln = nn.LayerNorm(self.hidden_dim2)
                self.fuse_e_ln = nn.LayerNorm(self.hidden_dim2)
                self.fuse_e_up = nn.Linear(self.hidden_dim // 2, self.hidden_dim2)
                self.fuse_gamma1 = nn.Parameter(torch.tensor(1.0))
                self.fuse_gamma2 = nn.Parameter(torch.tensor(1.0))
            else:
                self.fuse_h_ln = self.fuse_a_ln = self.fuse_e_ln = None
                self.fuse_e_up = None
                self.fuse_gamma1 = self.fuse_gamma2 = None
            self.token_feat_dim = (
                self.hidden_dim2 + self.hidden_dim // 2
                if self.eapatch_de_concat == 'on' else self.hidden_dim2
            )
        else:
            self.res_gamma = None
            self.res_ln = None
            self.attn_ln = None
            self.fuse_h_ln = self.fuse_a_ln = self.fuse_e_ln = None
            self.fuse_e_up = None
            self.fuse_gamma1 = self.fuse_gamma2 = None
            self.token_feat_dim = self.hidden_dim2

    def _preprocess_de_for_le(self, de_for_le):
        """LE-path only. de_for_le: [B,C,5] or [B,P,C,5]. Never call on de_raw / de_for_ge."""
        x = de_for_le
        if self.de_norm == 'bn' and self.de_bn is not None:
            shape = x.shape
            x = self.de_bn(x.reshape(-1, shape[-1])).reshape(shape)
        elif self.de_norm == 'ln' and self.de_ln is not None:
            x = self.de_ln(x)
        x = x + 1e-9  # LE copy only
        return x

    def _project_de_features(self, de_le, bs, num_node):
        """Apply clamp + filter_proj. Returns filter_features and expanded [B,P*C,H/2]."""
        if de_le.dim() == 3:
            # trial: [B,C,5] → broadcast later
            de_flat = de_le.reshape(bs * num_node, self.filter_feature_dim)
            if torch.isnan(de_flat).any() or torch.isinf(de_flat).any():
                print("Warning: NaN/inf in LE DE input, replacing with zeros")
                de_flat = torch.where(
                    torch.isnan(de_flat) | torch.isinf(de_flat),
                    torch.zeros_like(de_flat), de_flat,
                )
            if self.de_clamp == 'on':
                de_flat = torch.clamp(de_flat, min=-10, max=10)
            ff = self.filter_proj(de_flat)
            if torch.isnan(ff).any() or torch.isinf(ff).any():
                print("Warning: filter_proj output NaN/inf, using zeros")
                ff = torch.zeros_like(ff)
            ff = ff.view(bs, num_node, self.hidden_dim // 2)
            # expand to [B, P*C, H/2]
            ff_exp = ff.unsqueeze(1).repeat(1, self.num_patch, 1, 1).view(
                bs, self.num_patch * num_node, self.hidden_dim // 2
            )
            return ff, ff_exp
        # patch: [B,P,C,5]
        B, P, C, Fd = de_le.shape
        de_flat = de_le.reshape(B * P * C, Fd)
        if torch.isnan(de_flat).any() or torch.isinf(de_flat).any():
            print("Warning: NaN/inf in LE patch-DE input, replacing with zeros")
            de_flat = torch.where(
                torch.isnan(de_flat) | torch.isinf(de_flat),
                torch.zeros_like(de_flat), de_flat,
            )
        if self.de_clamp == 'on':
            de_flat = torch.clamp(de_flat, min=-10, max=10)
        ff = self.filter_proj(de_flat)
        if torch.isnan(ff).any() or torch.isinf(ff).any():
            print("Warning: filter_proj output NaN/inf, using zeros")
            ff = torch.zeros_like(ff)
        ff = ff.view(B, P, C, self.hidden_dim // 2)
        ff_exp = ff.view(B, P * C, self.hidden_dim // 2)
        return ff, ff_exp

    def forward(self, X, de_only=False):
        bs, num_node, time_length_origin = X.size()
        filter_features = None
        filter_features_expanded = None
        de_raw = None
        de_for_ge = None
        de_for_le = None
        de_patch = None
        self.last_x_ge = None
        self.last_de_raw = None
        self.last_de_for_ge = None
        self.last_de_for_le = None
        self.last_de_patch = None
        self.last_H_b = None
        self.last_attn_out = None
        self.last_Z_b = None
        attn_out_saved = None

        if self.use_eeg_filter:
            # Filter once; split immediately — GE gets trial DE only
            filtered = self.eeg_filter.filter_signal(X)  # [B,C,5,L]
            de_raw = EEGBandFilterOptimized.de_from_filtered(filtered)
            if torch.isnan(de_raw).any():
                print("Warning: de_raw contains NaN values")
            de_for_ge = de_raw  # alias — GE path; never mutate
            # Build x_ge for BandGraphGE (same layout as prior forward(return_windows))
            T = int(self.ge_num_windows) if self.ge_num_windows else 1
            if T <= 1:
                x_ge = de_raw.permute(0, 2, 1).unsqueeze(1).contiguous()
            else:
                B, C, Fb, L = filtered.shape
                assert L % T == 0, f'signal_length {L} not divisible by num_windows {T}'
                win = filtered.view(B, C, Fb, T, L // T)
                de_win = EEGBandFilterOptimized.de_from_filtered(win)
                x_ge = de_win.permute(0, 3, 2, 1).contiguous()
            self.last_x_ge = x_ge
            self.last_de_raw = de_raw
            self.last_de_for_ge = de_for_ge

            if EnAsEmbedding._ge_input_echo_n < 2:
                with torch.no_grad():
                    print(
                        f"[GE-INPUT] source=de_raw untouched=True "
                        f"checksum={float(de_for_ge.sum().item()):.10f} "
                        f"shape={tuple(de_for_ge.shape)} x_ge={tuple(x_ge.shape)}"
                    )
                EnAsEmbedding._ge_input_echo_n += 1

            # LE-only DE source
            if self.eapatch_de_level == 'patch':
                de_pc5p = EEGBandFilterOptimized.de_patch_from_filtered(
                    filtered, self.patch_size, self.num_patch, self.eapatch_de_window
                )  # [B,C,5,P]
                de_patch = de_pc5p.permute(0, 3, 1, 2).contiguous()  # [B,P,C,5]
                de_for_le = de_patch.clone()
                self.last_de_patch = de_patch
            else:
                de_for_le = de_raw.clone()
            self.last_de_for_le = de_for_le

            if de_only:
                return EnAsResult(
                    embedded=None, de_raw=de_raw, de_for_ge=de_for_ge, de_for_le=de_for_le,
                    de_patch=de_patch,
                )

            de_le = self._preprocess_de_for_le(de_for_le)
            self.last_de_for_le = de_le
            if de_patch is not None:
                self.last_de_patch = de_le  # preprocessed patch DE for MoE token source
            filter_features, filter_features_expanded = self._project_de_features(
                de_le, bs, num_node
            )

        # Patch embedding → H_b
        Xp = torch.reshape(X, [bs, num_node, self.num_patch, self.patch_size])
        Xp = torch.transpose(Xp, 1, 2)
        bs, tlen, num_node, dimension = Xp.size()
        if self.patch_embed == 'linear':
            # Legacy: treat patch_size samples as Conv1d "channels" with k=1 (= Linear)
            A_input = torch.reshape(Xp, [bs * tlen * num_node, dimension, 1])
            A_input_ = self.nonlin_map(A_input)
            A_input_ = torch.reshape(A_input_, [bs * tlen * num_node, -1])
        else:
            # I1 tconv: true temporal conv on time axis [N,1,T]
            xt = torch.reshape(Xp, [bs * tlen * num_node, 1, dimension])
            xt = self.patch_tconv(xt)
            if self.patch_conv_pool == 'avg':
                pooled = self.patch_pool_avg(xt).squeeze(-1)
            elif self.patch_conv_pool == 'max':
                pooled = self.patch_pool_max(xt).squeeze(-1)
            elif self.patch_conv_pool == 'std':
                # temporal std ≈ energy / envelope (translation-tolerant; DE-family)
                pooled = xt.std(dim=-1, unbiased=False)
            else:  # avg_max
                pooled = torch.cat(
                    [self.patch_pool_avg(xt).squeeze(-1), self.patch_pool_max(xt).squeeze(-1)],
                    dim=-1,
                )
            A_input_ = self.patch_tconv_proj(pooled)
        A_input_ = self.nonlin_map2(A_input_)
        if self.patch_activation is not None:
            A_input_ = self.patch_activation(A_input_)
        if self.patch_ln is not None:
            A_input_ = self.patch_ln(A_input_)
        original_features = torch.reshape(A_input_, [bs, tlen * num_node, self.hidden_dim2])
        self.last_H_b = original_features

        if self.use_eeg_filter and filter_features_expanded is not None:
            if self.eapatch_attn_dir == 'h2e':
                # Q=H, K/V=Ê → output lives in entropy value space
                Q = self.q_proj(original_features)
                K = self.k_proj(filter_features_expanded)
                V = self.v_proj(filter_features_expanded)
            else:
                # e2h (legacy): Q=Ê, K/V=H → output lives in patch value space
                Q = self.q_proj(filter_features_expanded)
                K = self.k_proj(original_features)
                V = self.v_proj(original_features)
            attn_out, _ = self.cross_attn(query=Q, key=K, value=V, need_weights=False)
            attn_out_saved = attn_out
            self.last_attn_out = attn_out

            if EnAsEmbedding._mag_print_n < EnAsEmbedding._MAG_PRINT_MAX:
                with torch.no_grad():
                    a_std = float(attn_out.std().item())
                    h_std = float(original_features.std().item())
                    ratio = a_std / (h_std + 1e-12)
                    print(f"[diag] mag attn_out.std={a_std:.6f} H_b.std={h_std:.6f} "
                          f"attn/H_b={ratio:.4f} residual={self.eapatch_residual} "
                          f"gamma={float(self.res_gamma.detach().cpu()):.4f} "
                          f"attn_norm={self.eapatch_attn_norm} "
                          f"attn_dir={self.eapatch_attn_dir} fuse={self.eapatch_fuse} "
                          f"de_level={self.eapatch_de_level}")
                    EnAsEmbedding._mag_print_n += 1

            if self.eapatch_fuse == 'tri':
                # Z = LN(H) + γ1·LN(attn) + γ2·LN(Ê·W_up)
                h_n = self.fuse_h_ln(original_features)
                a_n = self.fuse_a_ln(attn_out)
                e_n = self.fuse_e_ln(self.fuse_e_up(filter_features_expanded))
                output = h_n + self.fuse_gamma1 * a_n + self.fuse_gamma2 * e_n
                if (not self.training
                        and EnAsEmbedding._mag_print_n < EnAsEmbedding._MAG_PRINT_MAX + 2):
                    with torch.no_grad():
                        self.last_fuse_stats = {
                            'gamma1': float(self.fuse_gamma1.detach().cpu()),
                            'gamma2': float(self.fuse_gamma2.detach().cpu()),
                            'std_h': float(h_n.std().item()),
                            'std_a': float(a_n.std().item()),
                            'std_e': float(e_n.std().item()),
                        }
                        print(
                            f"[fuse-tri] γ1={self.last_fuse_stats['gamma1']:.4f} "
                            f"γ2={self.last_fuse_stats['gamma2']:.4f} "
                            f"std_h={self.last_fuse_stats['std_h']:.4f} "
                            f"std_a={self.last_fuse_stats['std_a']:.4f} "
                            f"std_e={self.last_fuse_stats['std_e']:.4f}"
                        )
            else:
                attn_for_res = self.attn_ln(attn_out) if self.attn_ln is not None else attn_out
                if self.eapatch_residual == 'on':
                    output = original_features + self.res_gamma * attn_for_res
                    if self.res_ln is not None:
                        output = self.res_ln(output)
                else:
                    output = attn_for_res

            if self.eapatch_de_concat == 'on':
                output = torch.cat([output, filter_features_expanded], dim=-1)

            if self.diag_eapatch:
                with torch.no_grad():
                    mode = 'train' if self.training else 'eval'
                    n_mode = (EnAsEmbedding._diag_train_n if self.training
                              else EnAsEmbedding._diag_eval_n)
                    if n_mode < EnAsEmbedding._DIAG_MAX_PER_MODE:
                        feat = output.size(-1)
                        _z = output.reshape(bs, tlen, num_node * feat)
                        ap_max = float(_z.std(dim=1).max().item())
                        ap_mean = float(_z.std(dim=1).mean().item())
                        rows_eq = torch.allclose(
                            output[:, 0 * num_node + 5, :],
                            output[:, 1 * num_node + 5, :], atol=1e-6)
                        _q = Q.reshape(bs, tlen, num_node, -1)
                        q_ap = float(_q.std(dim=1).max().item())
                        print(f"[diag] mode={mode} across_patch_std_max =", ap_max)
                        print(f"[diag] mode={mode} across_patch_std_mean =", ap_mean)
                        print(f"[diag] mode={mode} rows_equal =", bool(rows_eq))
                        print(f"[diag] mode={mode} Q_across_patch_std_max =", q_ap)
                        print(f"[diag] mode={mode} de_level={self.eapatch_de_level} "
                              f"attn_dir={self.eapatch_attn_dir} fuse={self.eapatch_fuse}")
                        if self.training:
                            EnAsEmbedding._diag_train_n += 1
                        else:
                            EnAsEmbedding._diag_eval_n += 1
                    if (self.training
                            and EnAsEmbedding._diag_train_n >= EnAsEmbedding._DIAG_MAX_PER_MODE
                            and EnAsEmbedding._diag_eval_n < EnAsEmbedding._DIAG_MAX_PER_MODE):
                        raise DiagEapatchDone("diag-eapatch: train samples collected; proceed to eval")
                    if (EnAsEmbedding._diag_train_n >= EnAsEmbedding._DIAG_MAX_PER_MODE
                            and EnAsEmbedding._diag_eval_n >= EnAsEmbedding._DIAG_MAX_PER_MODE):
                        raise DiagEapatchDone(
                            f"diag-eapatch done: train={EnAsEmbedding._diag_train_n} "
                            f"eval={EnAsEmbedding._diag_eval_n} batches"
                        )
        else:
            output = original_features

        feat = output.size(-1)
        output = output.reshape(bs, self.num_patch, self.num_node * feat)
        self.last_Z_b = output
        return EnAsResult(
            embedded=output,
            de_raw=de_raw,
            de_for_ge=de_for_ge,
            de_for_le=de_for_le if self.last_de_for_le is None else self.last_de_for_le,
            H_b=self.last_H_b,
            attn_out=attn_out_saved,
            Z_b=output,
            de_patch=self.last_de_patch,
        )
