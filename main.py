import os
import sys
import argparse
import numpy as np

from cross_validation import CrossValidation

os.chdir(sys.path[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notice', type=str, default='Specific Description of the Experiment')
    # parser.add_argument('--data-path', type=str, default='./feature/')
    parser.add_argument('--data-path', type=str, default='./data/features/')
  
    parser.add_argument('--faced-data-path', type=str, default='')
    parser.add_argument('--num-electrodes', type=int, default=32)
    parser.add_argument('--feature-length', type=int, default=512, help="deap:512,faced:500")
    # parser.add_argument('--label-type', type=str, default='V', choices=['A', 'V', 'D', 'L'])
    parser.add_argument('--label-type', '--label_type', dest='label_type', type=str, default='V', choices=['A', 'V', 'D', 'L',
                                                                         'NT','N', 'T', 'P','ALL'], 
                                                                         help='NT means remove the Neutral')
    parser.add_argument('--data-prepare', type=bool, default=False)
    # parser.add_argument('--data-prepare', type=bool, default=True)
    # parser.add_argument('--data-using', type=str, default='FACED', choices=['DEAP','FACED','OTH'])
    # parser.add_argument('--dataset', type=str, default='FACED', choices=['DEAP', 'FACED', 'P'])
    parser.add_argument('--data-using', type=str, default='DEAP', choices=['DEAP','FACED','OTH'])
    parser.add_argument('--dataset', type=str, default='DEAP', choices=['DEAP', 'FACED', 'P'])
    parser.add_argument('--subject_begin', type=int, default=0, help='If the programme is interrupted subject_begin can be quickly restarted')
    parser.add_argument('--subjects-FACED', type=int, default=40)
    parser.add_argument('--subjects-DEAP', type=int, default=32)
    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--combinedfeature-save-path', default='./feature/')
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--segment', type=int, default=2)
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=128, help='DEAP sampling rate (Hz)')
    parser.add_argument('--data-format', type=str, default='eeg')
    parser.add_argument('--pool', type=int, default=16)
    # parser.add_argument('--model', type=str, default='CrossGcnAtt', choices=['SimpleGCN', 'HGCN', 'EEGNet', 'CrossGcnAtt'])
    parser.add_argument('--model', type=str, default='moe', choices=['moe'])
    parser.add_argument('--dim-expand', type=bool, default=False, help="Expand data dim to adapte CNN/EEGNET model ")
    parser.add_argument('--edge-compute', type=str, default='COS', choices=['PLI', 'COR', 'COS'])
    parser.add_argument('--DE', type= bool, default=False, help='Compute the DE feature, b,c,F->b,c,5')
    # parser.add_argument('--load-path', default='./save/max-f1.pth')
    # parser.add_argument('--load-path-final', default='./save/max-f1.pth', help="if reproduce, set final_model.pth")
    parser.add_argument('--load-path', default='./save/max-f1.pth')
    parser.add_argument('--load-path-final', default='./save/max-f1.pth', help="if reproduce, set final_model.pth")    
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--data-norm', type=bool, default=False, help="normalize the raw data")
    #Training and Model Parameters
    parser.add_argument('--random-seed', type=int, default=666)
    parser.add_argument('--dense', type=str, default='full', choices=['full', 'sparse', 'phy'])
    parser.add_argument('--max-epoch', type=int, default=200, help="default max epoch 200")
    parser.add_argument('--early-stop-counter', type=int, default=20, help='early stop patience')
    parser.add_argument('--cts', type=bool, default=False, help="combine training switch")
    parser.add_argument('--patient-cmb', type=int, default=8)
    parser.add_argument('--max-epoch-cmb', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', '--learning_rate', dest='learning_rate', type=float, default=1e-3)
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=32)
    # parser.add_argument('--LS', type=bool, default=True, help="Label smoothing")
    parser.add_argument('--LS', type=bool, default=False, help="Label smoothing")
    parser.add_argument('--opt', type=str,default='AdamAutoDrop', choices=['Adam','SGD','AdamAutoDrop'])
    parser.add_argument('--LS-rate', type=float, default=0.1)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--T', type=int, default=64, help="window size for dynamic GNN")

# EnAs Embedding parameters
    parser.add_argument('--feature-embed-dim', type=int, default=64)
    parser.add_argument('--patch-size', '--patch_size', dest='patch_size', type=int, default=32)
    parser.add_argument('--num-patch', type=int, default=16)
    parser.add_argument('--dropout-rate', type=float, default=0.1)
    parser.add_argument('--use-eeg-filter', type=bool, default=True)
    parser.add_argument('--fs', type=int, default=128)
    # init patch Encoder
    parser.add_argument('--hidden-dim-conv1', '--hidden_dim_conv1', dest='hidden_dim_conv1', type=int, default=64)
    parser.add_argument('--hidden-dim-linear', '--hidden_dim_linear', dest='hidden_dim_linear', type=int, default=64)
    parser.add_argument('--num-heads-CR', type=int, default=8)    

    parser.add_argument('--num-experts', type=int, default=4)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--expert-dropout', type=float, default=0.15)
    parser.add_argument('--le-expert-arch', '--le_expert_arch', dest='le_expert_arch',
                        type=str, default='mlp', choices=['mlp', 'linear'],
                        help='LE path only; does not affect GE. mlp=2-layer ReLU (default); '
                             'linear=nn.Linear(D,C) no hidden/activation (ablation ≡ probe when E=k=1).')
    parser.add_argument('--loss-coef', type=float, default=1e-4)
    parser.add_argument('--expert-out', '--expert_out', dest='expert_out', type=str, default='logits',
                        choices=['logits', 'softmax'],
                        help='P1: expert output space. logits=paper (fuse before CE softmax); softmax=legacy A/B')
    parser.add_argument('--class-weight', '--class_weight', dest='class_weight', type=str, default='none',
                        choices=['none', 'balanced'],
                        help='P2: CE class weights. balanced=inv freq on current train fold, normalized')
    parser.add_argument('--validation-type', type=str, default='n_fold', choices=['n_fold'])
    # P0 experiment control switches
    parser.add_argument('--subject-list', '--subject_list', dest='subject_list', type=str, default=None,
                        help='Comma-separated subject indices, e.g. "0,1,2,3,4,5,6". Overrides full subject range when set.')
    parser.add_argument('--outer-folds', '--outer_folds', dest='outer_folds', type=str, default=None,
                        help='Comma-separated outer fold indices to run, e.g. "0,1,2". KFold split unchanged; others skipped.')
    parser.add_argument('--branch', type=str, default='le_ge', choices=['le', 'le_ge', 'ge'],
                        help='le=LE-only; le_ge=LE+GE fusion; ge=GE-only')
    parser.add_argument('--fusion-alpha', '--fusion_alpha', dest='fusion_alpha', type=float, default=0.9,
                        help='le_ge train-time fuse: y=alpha*LE+(1-alpha)*GE (LE weight)')
    parser.add_argument('--ge-fuse-space', '--ge_fuse_space', dest='ge_fuse_space', type=str,
                        default='logit', choices=['logit', 'prob'],
                        help='B18: fuse GE as logits (paper) or softmax probs (legacy)')
    parser.add_argument('--fusion-scale', '--fusion_scale', dest='fusion_scale', type=str,
                        default='none', choices=['none', 'ln', 'temp'],
                        help='Scale y_glob before fuse: none / LayerNorm(K) / learnable temp')
    parser.add_argument('--fusion-detach-scale', '--fusion_detach_scale',
                        dest='fusion_detach_scale', type=str, default='zscore',
                        choices=['off', 'zscore', 'frozen'],
                        help='F3: detach μ/σ z-score both branches before fuse (blocks amp compensation)')
    parser.add_argument('--fusion-alpha-mode', '--fusion_alpha_mode', dest='fusion_alpha_mode',
                        type=str, default='learn_trial',
                        choices=['fixed', 'learn_scalar', 'learn_trial'],
                        help='F3: fixed α / learnable scalar / trial-adaptive α_b (paper §3.5)')
    parser.add_argument('--fusion-alpha-min', '--fusion_alpha_min', dest='fusion_alpha_min',
                        type=float, default=0.3,
                        help='Hard lower bound on LE weight (learn_* modes)')
    parser.add_argument('--fusion-alpha-init', '--fusion_alpha_init', dest='fusion_alpha_init',
                        type=float, default=0.9,
                        help='Initial / prior target α (learn_*); step-0 α equals this')
    parser.add_argument('--fusion-alpha-tau', '--fusion_alpha_tau', dest='fusion_alpha_tau',
                        type=float, default=1.0,
                        help='Temperature on α logits (learn_*)')
    parser.add_argument('--fusion-alpha-warmup', '--fusion_alpha_warmup', dest='fusion_alpha_warmup',
                        type=int, default=5,
                        help='Force α=1.0 for first N epochs (learn_*); 0 disables')
    parser.add_argument('--fusion-alpha-lr-scale', '--fusion_alpha_lr_scale',
                        dest='fusion_alpha_lr_scale', type=float, default=0.3,
                        help='α-head lr = main_lr × this')
    parser.add_argument('--fusion-alpha-prior', '--fusion_alpha_prior', dest='fusion_alpha_prior',
                        type=float, default=1e-3,
                        help='λ for mean((α_b − α_init)^2); 0 disables')
    parser.add_argument('--dump-alpha', '--dump_alpha', dest='dump_alpha',
                        type=str, default='off', choices=['on', 'off'],
                        help='Accumulate per-batch α_b (non-warmup) and dump summary npz; '
                             'off = bit-identical default')
    parser.add_argument('--aux-branch-ce-loc', '--aux_branch_ce_loc', dest='aux_branch_ce_loc',
                        type=float, default=1.0,
                        help='Scheme A: λ_loc · CE(y_loc, y); 0 disables (bit-identical default)')
    parser.add_argument('--aux-branch-ce-glob', '--aux_branch_ce_glob', dest='aux_branch_ce_glob',
                        type=float, default=1.0,
                        help='Scheme A: λ_glob · CE(y_glob, y); 0 disables (bit-identical default)')
    parser.add_argument('--branch-pretrain-epochs', '--branch_pretrain_epochs',
                        dest='branch_pretrain_epochs', type=int, default=0,
                        help='Scheme C: first N epochs CE(y_loc)+CE(y_glob) no fuse; 0 disables')
    parser.add_argument('--alpha-mode', '--alpha_mode', dest='alpha_mode', type=str,
                        default='fixed', choices=['fixed', 'val_select'],
                        help='fixed=use fusion_alpha; val_select=pick α* on val logits (never test)')
    parser.add_argument('--alpha-objective', '--alpha_objective', dest='alpha_objective', type=str,
                        default='auc', choices=['auc', 'macro_f1', 'acc_f1_mean'],
                        help='Objective for val_select α* (default auc)')
    parser.add_argument('--run-tag', '--run_tag', dest='run_tag', type=str, default='',
                        help='Tag for log / result file naming')
    parser.add_argument('--diag-eapatch', '--diag_eapatch', dest='diag_eapatch', type=str,
                        default='off', choices=['on', 'off'],
                        help='Read-only: print Z_b / Q patch-variance after cross-attn (default off)')
    # P4: EAPatch residual (scheme B) + B13 patch nonlinearity
    parser.add_argument('--eapatch-residual', '--eapatch_residual', dest='eapatch_residual',
                        type=str, default='on', choices=['on', 'off'],
                        help='P4/B: Z=LN(H+gamma*attn). off=legacy replace (attn_out only)')
    parser.add_argument('--eapatch-res-norm', '--eapatch_res_norm', dest='eapatch_res_norm',
                        type=str, default='ln', choices=['none', 'ln'],
                        help='Norm after residual add (default ln)')
    parser.add_argument('--eapatch-res-gamma', '--eapatch_res_gamma', dest='eapatch_res_gamma',
                        type=str, default='learnable', choices=['fixed', 'learnable'],
                        help='Residual scale gamma: fixed=1.0 or learnable scalar init 1.0')
    parser.add_argument('--patch-act', '--patch_act', dest='patch_act',
                        type=str, default='gelu', choices=['none', 'gelu', 'relu'],
                        help='B13: nonlinearity after nonlin_map2 (paper phi)')
    parser.add_argument('--patch-norm', '--patch_norm', dest='patch_norm',
                        type=str, default='ln', choices=['none', 'ln'],
                        help='B13: LayerNorm after patch act (default ln)')
    parser.add_argument('--smoke', action='store_true', default=False,
                        help='Mark run as smoke test')
    parser.add_argument('--ge-layout', '--ge_layout', dest='ge_layout',
                        type=str, default='legacy', choices=['legacy', 'tb_T'],
                        help='GE input layout: legacy=[B,1,C,F] channel-major; '
                             'tb_T=[B,1,F,C] band-as-node (paper T_b^T)')
    parser.add_argument('--adj-nonneg', '--adj_nonneg', dest='adj_nonneg',
                        type=str, default='none', choices=['none', 'softplus', 'clamp'],
                        help='Nonneg constraint for GE adjacency before DAD norm')
    parser.add_argument('--ge-pool', '--ge_pool', dest='ge_pool',
                        type=str, default='cls', choices=['cls', 'mean'],
                        help='GE graph pooling: cls (legacy) or mean (paper)')
    # ---- New BandGraphGE (--ge-impl bandgraph) ----
    parser.add_argument('--ge-impl', '--ge_impl', dest='ge_impl',
                        type=str, default='bandgraph',
                        choices=['bandgraph', 'linear_probe'],
                        help='GE implementation: bandgraph=band-node MoE; '
                             'linear_probe=flatten DE→Linear(160,2)')
    parser.add_argument('--ge-num-windows', '--ge_num_windows', dest='ge_num_windows',
                        type=int, default=1, choices=[1, 2, 4],
                        help='T time windows for GE DE (filter once on full L then split). '
                             'T=1:4s, T=2:2s, T=4:1s. T≥8 forbidden: delta 0.5–4Hz '
                             'has <1 period in 0.5s windows → noisy variance.')
    parser.add_argument('--ge-num-experts', '--ge_num_experts', dest='ge_num_experts',
                        type=int, default=5,
                        help='E_g graph experts (default 5=one per band; E_g>1 required for expressivity)')
    parser.add_argument('--ge-num-layers', '--ge_num_layers', dest='ge_num_layers',
                        type=int, default=1, choices=[1, 2],
                        help='Number of BandGraphGE blocks')
    parser.add_argument('--ge-hidden', '--ge_hidden', dest='ge_hidden',
                        type=int, default=32, help='GE hidden / C_out (default=C=32 for residual)')
    parser.add_argument('--ge-heads', '--ge_heads', dest='ge_heads',
                        type=int, default=4, help='Band MHSA heads')
    parser.add_argument('--ge-dim-head', '--ge_dim_head', dest='ge_dim_head',
                        type=int, default=16, help='Band MHSA dim per head (inner=heads*dim_head)')
    parser.add_argument('--ge-dropout', '--ge_dropout', dest='ge_dropout',
                        type=float, default=0.1,
                        help='GE dropout (default 0.1; do NOT use 0.5 with n_train≈360)')
    parser.add_argument('--ge-node-norm', '--ge_node_norm', dest='ge_node_norm',
                        type=str, default='ln', choices=['none', 'ln'],
                        help='Per-node LayerNorm on C dims before graph (default ln)')
    parser.add_argument('--ge-temporal', '--ge_temporal', dest='ge_temporal',
                        type=str, default='mean', choices=['mean', 'attn', 'conv'],
                        help='Temporal aggregation over T windows after node mean-pool')
    parser.add_argument('--ge-adj-nonneg', '--ge_adj_nonneg', dest='ge_adj_nonneg',
                        type=str, default='softplus', choices=['softplus', 'abs'],
                        help='Nonneg for BandGraphGE adjacency (default softplus)')
    parser.add_argument('--ge-routing', '--ge_routing', dest='ge_routing',
                        type=str, default='st_hard', choices=['st_hard', 'soft'],
                        help='Expert routing: st_hard=straight-through argmax; soft=gamma weighted')
    parser.add_argument('--ge-delta-feat', '--ge_delta_feat', dest='ge_delta_feat',
                        type=str, default='off', choices=['on', 'off'],
                        help='Concat ΔDE across adjacent windows as extra node feats (default off)')
    parser.add_argument('--ge-stage', '--ge_stage', dest='ge_stage',
                        type=str, default='l3', choices=['l1', 'l2', 'l3', 'full'],
                        help='Validation ladder: l1=LN+pool+Linear; l2=+GCN E=1; '
                             'l3=+E_g experts ST; full=+MHSA+node-conv')
    parser.add_argument('--ge-check-finite', '--ge_check_finite', dest='ge_check_finite',
                        type=str, default='off', choices=['on', 'off'],
                        help='Optional assert torch.isfinite in BandGraphGE forward')
    parser.add_argument('--ge-use-mhsa', '--ge_use_mhsa', dest='ge_use_mhsa',
                        type=str, default='on', choices=['auto', 'on', 'off'],
                        help='L4 ablation: force MHSA on/off (auto=follow --ge-stage)')
    parser.add_argument('--ge-use-node-conv', '--ge_use_node_conv', dest='ge_use_node_conv',
                        type=str, default='off', choices=['auto', 'on', 'off'],
                        help='L4 ablation: force node-Conv1d on/off (auto=follow --ge-stage)')
    parser.add_argument('--ge-self-loop', '--ge_self_loop', dest='ge_self_loop',
                        type=str, default='after_softplus',
                        choices=['after_softplus', 'before_softplus'],
                        help='When to add I: after_softplus (default; diag≈1.693) or '
                             'before_softplus (I then softplus)')
    parser.add_argument('--ge-wg-init-std', '--ge_wg_init_std', dest='ge_wg_init_std',
                        type=float, default=0.02,
                        help='W_g gate init Normal(0, std); default 0.02')
    parser.add_argument('--ge-diag-routing', '--ge_diag_routing', dest='ge_diag_routing',
                        type=str, default='off', choices=['on', 'off'],
                        help='Accumulate node×expert routing counts + I(v;e) report')
    parser.add_argument('--le-routing', '--le_routing', dest='le_routing',
                        type=str, default='patch', choices=['trial', 'patch'],
                        help='P5: trial=flat P*D token (pre-P1); patch=(b,p) tokens + mean_p')
    # ---- LE path only (must NOT affect GE / de_raw) ----
    parser.add_argument('--de-norm', '--de_norm', dest='de_norm',
                        type=str, default='none', choices=['none', 'bn', 'ln'],
                        help='LE path only; does not affect GE. Normalize de_for_le before filter_proj.')
    parser.add_argument('--filter-proj-gain', '--filter_proj_gain', dest='filter_proj_gain',
                        type=float, default=0.1,
                        help='LE path only; does not affect GE. Xavier gain for filter_proj (default 0.1).')
    parser.add_argument('--de-clamp', '--de_clamp', dest='de_clamp',
                        type=str, default='on', choices=['on', 'off'],
                        help='LE path only; does not affect GE. Clamp de_for_le to [-10,10] before proj.')
    parser.add_argument('--eapatch-attn-norm', '--eapatch_attn_norm', dest='eapatch_attn_norm',
                        type=str, default='none', choices=['none', 'ln'],
                        help='LE path only; does not affect GE. LN attn_out before residual add.')
    parser.add_argument('--eapatch-res-gamma-init', '--eapatch_res_gamma_init',
                        dest='eapatch_res_gamma_init', type=float, default=1.0,
                        help='LE path only; does not affect GE. Initial value of residual gamma.')
    parser.add_argument('--eapatch-de-concat', '--eapatch_de_concat', dest='eapatch_de_concat',
                        type=str, default='off', choices=['off', 'on'],
                        help='LE path only; does not affect GE. Concat DE proj onto each patch token.')
    parser.add_argument('--le-token-proj', '--le_token_proj', dest='le_token_proj',
                        type=int, default=0,
                        help='LE path only; does not affect GE. If >0, Linear(token_dim→N) before gate/experts.')
    parser.add_argument('--eapatch-de-level', '--eapatch_de_level', dest='eapatch_de_level',
                        type=str, default='trial', choices=['trial', 'patch'],
                        help='LE path only; does not affect GE. '
                             'trial=broadcast trial-level DE (legacy); '
                             'patch=per-patch DE via overlapping windows centered on each patch.')
    parser.add_argument('--eapatch-de-window', '--eapatch_de_window', dest='eapatch_de_window',
                        type=int, default=128, choices=[128, 256],
                        help='LE path only; does not affect GE. DE estimation window (samples) for '
                             '--eapatch-de-level patch. Default 128=1s@128Hz (≥0.5 period of delta '
                             '0.5Hz). Patch positions stay at stride=patch_size; short patches alone '
                             'cannot estimate delta.')
    parser.add_argument('--le-token-source', '--le_token_source', dest='le_token_source',
                        type=str, default='z',
                        choices=['z', 'de_patch', 'concat', 'split_concat'],
                        help='LE path only; does not affect GE. Token for MoE: z=EAPatch output; '
                             'de_patch=flatten per-patch DE (160-d); concat=[proj(z)]⊕de_patch; '
                             'split_concat=LN(Linear(z,64))⊕LN(Linear(de,64)).')
    parser.add_argument('--le-token-norm', '--le_token_norm', dest='le_token_norm',
                        type=str, default='none', choices=['none', 'ln', 'bn'],
                        help='LE path only; does not affect GE. Norm tokens before gate/experts.')
    parser.add_argument('--eapatch-attn-dir', '--eapatch_attn_dir', dest='eapatch_attn_dir',
                        type=str, default='e2h', choices=['e2h', 'h2e'],
                        help='LE path only; does not affect GE. Cross-attn direction: '
                             'e2h=Q from entropy K/V from patch (legacy); '
                             'h2e=Q from patch K/V from entropy (entropy enters output).')
    parser.add_argument('--eapatch-fuse', '--eapatch_fuse', dest='eapatch_fuse',
                        type=str, default='residual', choices=['residual', 'tri'],
                        help='LE path only; does not affect GE. residual=H+γ·attn; '
                             'tri=LN(H)+γ1·LN(attn)+γ2·LN(Ê·W_up).')
    # I1: true temporal Conv1d patch embed (LE path only; does not affect GE)
    parser.add_argument('--patch-embed', '--patch_embed', dest='patch_embed',
                        type=str, default='tconv', choices=['linear', 'tconv'],
                        help='LE path only; does not affect GE. linear=legacy k=1 Linear(32→H); '
                             'tconv=true temporal Conv1d on patch time axis.')
    parser.add_argument('--patch-conv-kernel', '--patch_conv_kernel', dest='patch_conv_kernel',
                        type=int, default=7, choices=[5, 7, 15],
                        help='LE path only; does not affect GE. Temporal conv kernel for --patch-embed tconv.')
    parser.add_argument('--patch-conv-chans', '--patch_conv_chans', dest='patch_conv_chans',
                        type=int, default=32,
                        help='LE path only; does not affect GE. Temporal conv channels.')
    parser.add_argument('--patch-conv-layers', '--patch_conv_layers', dest='patch_conv_layers',
                        type=int, default=1, choices=[1, 2],
                        help='LE path only; does not affect GE. Number of temporal conv blocks.')
    parser.add_argument('--patch-conv-pool', '--patch_conv_pool', dest='patch_conv_pool',
                        type=str, default='avg', choices=['avg', 'max', 'avg_max', 'std'],
                        help='LE path only; does not affect GE. Temporal pool after tconv '
                             '(avg_max=mean∥max envelope; std=temporal std / energy-like).')
    # I4: patch positional embedding (LE path only; does not affect GE)
    parser.add_argument('--le-pos-embed', '--le_pos_embed', dest='le_pos_embed',
                        type=str, default='none', choices=['none', 'learn', 'sincos'],
                        help='LE path only; does not affect GE. Add patch-position embedding to z_{b,p}.')
    # Measurement head for EAPatch (LE path only; does not affect GE)
    parser.add_argument('--eapatch-head', '--eapatch_head', dest='eapatch_head',
                        type=str, default='moe', choices=['linear', 'mlp', 'moe'],
                        help='LE path only; does not affect GE. linear/mlp=mean_P(Z_b)→clf '
                             '(bypass MoE to measure EAPatch); moe=sparse-gated FFN (status quo).')
    parser.add_argument('--thr-objective', '--thr_objective', dest='thr_objective',
                        type=str, default='acc_f1_mean',
                        choices=['argmax', 'macro_f1', 'acc_f1_mean', 'balanced_acc'],
                        help='LE path only; does not affect GE. Val-set threshold selection '
                             'objective (never touch test). argmax=fixed t=0.5.')
    parser.add_argument('--le-probe-dump', '--le_probe_dump', dest='le_probe_dump',
                        type=str, default='off', choices=['on', 'off'],
                        help='LE path only; does not affect GE. Side-channel dump A/B/C/D (no compute change).')
    parser.add_argument('--le-probe-dir', '--le_probe_dir', dest='le_probe_dir',
                        type=str, default='./save/le_probe',
                        help='Directory for --le-probe-dump tensors (LE path only; does not affect GE).')
    parser.add_argument('--le-probe-keys', '--le_probe_keys', dest='le_probe_keys',
                        type=str, default='de_raw,H_b,attn_out,Z_b,de_patch',
                        help='Comma-separated tensors to dump with --le-probe-dump '
                             '(LE path only; use H_b alone to save disk).')
    parser.add_argument('--split-level', '--split_level', dest='split_level',
                        type=str, default='segment', choices=['segment', 'trial'],
                        help='CV unit: segment=KFold on 600 segs (legacy/leak-risk); '
                             'trial=KFold on 40 trials then expand (paper protocol). '
                             'Default segment keeps historical numbers comparable.')
# Reproduce the result using the saved model
    parser.add_argument('--reproduce', type= bool, default=False)
    args = parser.parse_args()
    # Ensure data_path ends with / (cross_validation does string concat)
    if args.data_path and not args.data_path.endswith(('/', '\\')):
        args.data_path = args.data_path + '/'
    args.cmdline = 'python ' + ' '.join(sys.argv)
    print(
        f"[CONFIG-ECHO] main args.ge_impl={getattr(args,'ge_impl',None)} "
        f"ge_num_windows={getattr(args,'ge_num_windows',None)} "
        f"ge_stage={getattr(args,'ge_stage',None)} "
        f"ge_num_experts={getattr(args,'ge_num_experts',None)} "
        f"ge_layout={getattr(args,'ge_layout',None)!r} "
        f"adj_nonneg={getattr(args,'adj_nonneg',None)} ge_pool={getattr(args,'ge_pool',None)} "
        f"branch={getattr(args,'branch',None)} fs={getattr(args,'fs',None)} "
        f"use_eeg_filter={getattr(args,'use_eeg_filter',None)} "
        f"num_heads_CR={getattr(args,'num_heads_CR',None)} "
        f"dropout_rate={getattr(args,'dropout_rate',None)} "
        f"expert_dropout={getattr(args,'expert_dropout',None)} "
        f"split_level={getattr(args,'split_level',None)}"
    )
    # Hard assert for T (argparse choices already limit, but catch underscore overrides)
    _nw = int(getattr(args, 'ge_num_windows', 4) or 4)
    assert _nw in (1, 2, 4), (
        f'--ge-num-windows must be in {{1,2,4}}, got {_nw}. '
        f'T≥8 forbidden: delta band (0.5–4Hz) has <1 period in 0.5s windows.'
    )
    # Reset EAPatch diag counters for a clean process-level probe
    if str(getattr(args, 'diag_eapatch', 'off')).lower() == 'on':
        try:
            from moe import EnAsEmbedding
            EnAsEmbedding._diag_train_n = 0
            EnAsEmbedding._diag_eval_n = 0
            EnAsEmbedding._mag_print_n = 0
        except Exception:
            pass

    tag = str(getattr(args, 'run_tag', '') or '').strip()
    if tag:
        args.save_path = os.path.join('./save', tag) + '/'
        ckpt_name = os.path.basename(args.load_path) or 'max-f1.pth'
        args.load_path = os.path.join(args.save_path, ckpt_name)
        args.load_path_final = args.load_path

    validator = CrossValidation(args)
    if args.data_using != 'DEAP':
        raise ValueError('This release trains on DEAP only. Set --data-using DEAP.')
    if args.subject_list is not None and str(args.subject_list).strip() != '':
        subject = np.array([int(x.strip()) for x in str(args.subject_list).split(',') if x.strip() != ''])
    else:
        subject = np.arange(32)
    validator.n_fold_CV(subject=subject)
