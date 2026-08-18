from utils import *
import torch.nn as nn
import os.path as osp
import numpy as np

try:
    from moe import DiagEapatchDone
except Exception:
    DiagEapatchDone = None

CUDA = torch.cuda.is_available()


# def train_one_epoch(data_loader, net, loss_fn, optimizer):
#     net.train()
#     tl = Averager()
#     pred_train = []
#     act_train = []
#     for i, (x_batch, y_batch) in enumerate(data_loader):
#         if CUDA:
#             x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
#         out = net(x_batch)
#         loss = loss_fn(out, y_batch)
#         _, pred = torch.max(out, 1)
#         tl.add(loss)
#         pred_train.extend(pred.data.tolist())
#         act_train.extend(y_batch.data.tolist())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     return tl.item(), pred_train, act_train

def train_one_epoch(args, data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    score_train = []  # softmax[:,1] for train AUC
    nan_batch = 0
    expert_frac_sum = None
    expert_frac_n = 0
    aux_sum = 0.0
    aux_n = 0
    div_sum = 0.0
    div_n = 0
    alpha_mean_sum = 0.0
    alpha_std_sum = 0.0
    alpha_min_seen = 1.0
    alpha_max_seen = 0.0
    alpha_n = 0
    le_std_sum = 0.0
    ge_std_sum = 0.0
    scale_n = 0
    dump_alpha = str(getattr(args, 'dump_alpha', 'off')).lower() == 'on'
    if dump_alpha and not hasattr(net, '_alpha_dump_chunks'):
        net._alpha_dump_chunks = []

    for data in data_loader:
        if CUDA:
            data = data.cuda()
            x_batch = data
            y_batch = data.y.long()
            # optimizer.zero_grad()
            if args.dim_expand :
                x_batch = x_batch.x.unsqueeze(1)
            if args.model == 'EEGNet' :
                x_batch = x_batch.x.unsqueeze(1)           
            if args.model == 'DGCNN' :
                x_batch = x_batch.x.view(-1,args.num_electrodes,args.feature_length)
            if args.model == 'ArjunViT':
                x_batch = x_batch.x.view(-1,args.num_electrodes,args.feature_length)            
            if args.model =='ATCNet':
                x_batch = x_batch.x.view(-1,args.num_electrodes,args.feature_length)
                x_batch = x_batch.unsqueeze(1)

            if args.model == 'MOGE':
                x_batch = x_batch.x.unsqueeze(1)
                # print(x_batch.shape)

            if args.model == 'RGNN':
                x_batch = x_batch
                # x_batch = x_batch[:,:,0:4]

            if args.model == 'TSCeption':
                x_batch = x_batch.x.unsqueeze(1) 

            if args.model == 'PGCN':
                x_batch = x_batch.x.view(-1,args.num_electrodes,5)

            if args.model == 'FC_STGNN' or args.model == 'FocalMoE' or args.model == 'moe':

                if isinstance(x_batch.x, list):
                    x_batch = torch.stack([torch.tensor(x, dtype=torch.float32) for x in x_batch.x])
                    if CUDA:
                        x_batch = x_batch.cuda()
                else:
                    x_batch = x_batch.x
                    if CUDA:
                        x_batch = x_batch.cuda()
                        
                # print("Original x_batch shape:", x_batch.shape)
                batch_size = x_batch.size(0)
                total_size = x_batch.numel()
                # expected_size = batch_size * args.num_electrodes * 5
                # if total_size != expected_size:
                #     print(f"Warning: total_size ({total_size}) != expected_size ({expected_size})")
                x_batch = x_batch.view(-1,args.num_electrodes,args.feature_length)
            if args.model == 'HierCorrPool':
                x_batch = x_batch.x.view(-1,args.num_electrodes,args.feature_length)

            if args.model == 'SoftShapeNet':
                if isinstance(x_batch.x, list):
                    x_batch = torch.stack([torch.tensor(x, dtype=torch.float32) for x in x_batch.x])
                    if CUDA:
                        x_batch = x_batch.cuda()
                else:
                    x_batch = x_batch.x
                    if CUDA:
                        x_batch = x_batch.cuda()
                x_batch = x_batch.view(-1,args.num_electrodes,args.feature_length)
                
            x_batch = x_batch.to('cuda')
            try:
                out = net(x_batch)
            except Exception as e:
                if DiagEapatchDone is not None and isinstance(e, DiagEapatchDone):
                    print(f'[diag-eapatch] early stop train_one_epoch: {e}')
                    break
                raise
            
            if args.model == 'moe' and isinstance(out, tuple):
                logits, aux_loss = out
                lam_loc = float(getattr(args, 'aux_branch_ce_loc', 0.0) or 0.0)
                lam_glob = float(getattr(args, 'aux_branch_ce_glob', 0.0) or 0.0)
                in_pretrain = bool(getattr(net, '_branch_pretrain', False))
                y_loc_live = getattr(net, 'last_y_loc_live', None)
                y_glob_live = getattr(net, 'last_y_glob_live', None)
                if in_pretrain and y_loc_live is not None and y_glob_live is not None:
                    # Scheme C: independent CE on both branches, no fuse loss
                    loss = (loss_fn(y_loc_live, y_batch)
                            + loss_fn(y_glob_live, y_batch)
                            + args.loss_coef * aux_loss)
                    with torch.no_grad():
                        logits = 0.5 * (y_loc_live.detach() + y_glob_live.detach())
                else:
                    main_loss = loss_fn(logits, y_batch)
                    loss = main_loss + args.loss_coef * aux_loss
                    # Scheme A: deep supervision on branches (default λ=0 → no-op)
                    if lam_loc > 0.0 and y_loc_live is not None:
                        loss = loss + lam_loc * loss_fn(y_loc_live, y_batch)
                    if lam_glob > 0.0 and y_glob_live is not None:
                        loss = loss + lam_glob * loss_fn(y_glob_live, y_batch)
                # F3: weak prior pulling α_b toward α_init
                alpha_prior = getattr(net, 'last_alpha_prior_loss', None)
                if (not in_pretrain) and alpha_prior is not None and torch.is_tensor(alpha_prior):
                    loss = loss + alpha_prior
                _, pred = torch.max(logits, 1)
                probs = torch.softmax(logits.detach(), dim=1)[:, 1]
                if getattr(net, 'last_expert_frac', None) is not None:
                    ef = net.last_expert_frac
                    expert_frac_sum = ef if expert_frac_sum is None else expert_frac_sum + ef
                    expert_frac_n += 1
                if getattr(net, 'last_aux_loss', None) is not None:
                    aux_sum += float(net.last_aux_loss)
                    aux_n += 1
                if getattr(net, 'last_patch_diversity', None) is not None:
                    div_sum += float(net.last_patch_diversity)
                    div_n += 1
                if getattr(net, 'last_alpha_stats', None) is not None:
                    st = net.last_alpha_stats
                    alpha_mean_sum += float(st.get('mean', 0.0))
                    alpha_std_sum += float(st.get('std', 0.0))
                    alpha_min_seen = min(alpha_min_seen, float(st.get('min', 1.0)))
                    alpha_max_seen = max(alpha_max_seen, float(st.get('max', 0.0)))
                    alpha_n += 1
                    # Optional dump: per-sample α_b when not in warmup force
                    if dump_alpha and not getattr(net, '_fusion_force_le', False):
                        a = getattr(net, 'last_alpha', None)
                        if a is not None and torch.is_tensor(a):
                            net._alpha_dump_chunks.append(a.detach().float().cpu().numpy().ravel())
                if getattr(net, 'last_ge_scale', None) is not None:
                    gs = net.last_ge_scale
                    le_std_sum += float(gs.get('le_std', 0.0))
                    ge_std_sum += float(gs.get('ge_std', 0.0))
                    scale_n += 1
            else:
                loss = loss_fn(out, y_batch)
                _, pred = torch.max(out, 1)
                probs = torch.softmax(out.detach(), dim=1)[:, 1]
            # P0: NaN/Inf guard — skip non-finite batches
            optimizer.zero_grad()
            if not torch.isfinite(loss).all():
                nan_batch += 1
                continue
            tl.add(float(loss.detach().mean().item()))
            pred_train.extend(pred.data.tolist())
            act_train.extend(y_batch.data.tolist())
            score_train.extend(probs.cpu().tolist())
            # Ensure scalar loss for backward when aux was 0-dim or 1-dim
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

    if nan_batch > 0:
        print(f'[NaN guard] skipped {nan_batch} non-finite loss batch(es) this epoch')
    if expert_frac_n > 0:
        mean_frac = (expert_frac_sum / expert_frac_n).tolist()
        max_frac = max(mean_frac)
        div_s = f' patch_experts_per_trial={div_sum/max(div_n,1):.2f}' if div_n else ''
        print(f'[MoE] expert_token_frac={["%.3f"%x for x in mean_frac]} max={max_frac:.3f} '
              f'aux={aux_sum/max(aux_n,1):.4f} nan_batch={nan_batch}{div_s}')
        if max_frac > 0.90:
            print(f'[WARN] expert collapse: one expert took >90% tokens')
    if alpha_n > 0:
        print(
            f'[fusion-α] mean={alpha_mean_sum/alpha_n:.4f} std={alpha_std_sum/alpha_n:.4f} '
            f'min={alpha_min_seen:.4f} max={alpha_max_seen:.4f} '
            f'(forced={getattr(net, "_fusion_force_le", False)})'
        )
    if scale_n > 0:
        print(
            f'[fusion-scale] y_loc.std={le_std_sum/scale_n:.4f} '
            f'y_glob.std={ge_std_sum/scale_n:.4f}'
        )
    train_auc = safe_roc_auc(act_train, score_train)
    loss_item = float(tl.item()) if not hasattr(tl.item(), 'detach') else float(tl.item())
    # Averager may hold 0-dim tensors after MoE aux; coerce to Python float
    try:
        loss_item = float(loss_item)
    except Exception:
        loss_item = float(np.asarray(tl.item()).reshape(-1)[0])
    auc_s = f'{train_auc:.4f}' if train_auc == train_auc else 'nan'
    print(f'[train-epoch] loss={loss_item:.4f} AUC={auc_s}')
    return loss_item, pred_train, act_train, train_auc


def predict(args, data_loader, net, loss_fn, subject=None, fold=None, log_tag='eval'):
    """
    Returns loss, preds, acts, score_pos, branch_dict.
    branch_dict is None or {'le':[N,K], 'ge':[N,K]} (pre-fuse scaled) for le_ge.
    """
    net.eval()
    pred_val = []
    act_val = []
    score_val = []
    le_logits_acc = []
    ge_logits_acc = []
    logits_acc = []
    alpha_acc = []
    vl = Averager()
    do_probe = str(getattr(args, 'le_probe_dump', 'off')).lower() == 'on'
    probe_acc = {
        'de_raw': [], 'H_b': [], 'attn_out': [], 'Z_b': [], 'de_patch': [], 'y': []
    } if do_probe else None
    with torch.no_grad():
        for data in data_loader:
            if CUDA:
                data = data.cuda()
                x_batch = data
                y_batch = data.y.long()
                if args.dim_expand :
                    x_batch = x_batch.x.unsqueeze(1)
                # if args.model == 'DGCNN':
                #     x_batch = x_batch.x.view(-1,args.num_electrodes,args.feature_length)
                if args.model == 'DGCNN' :
                    x_batch = x_batch.x.view(-1,args.num_electrodes,args.feature_length)
                if args.model == 'ArjunViT':
                    x_batch = x_batch.x.view(-1,args.num_electrodes,args.feature_length)
                if args.model =='ATCNet':
                    x_batch = x_batch.x.view(-1,args.num_electrodes,args.feature_length)
                    x_batch = x_batch.unsqueeze(1)
                if args.model == 'EEGNet' :
                    x_batch = x_batch.x.unsqueeze(1)     

                if args.model == 'MOGE':
                    x_batch = x_batch.x.unsqueeze(1)

                if args.model == 'RGNN':
                    x_batch = x_batch

                if args.model == 'TSCeption':
                    x_batch = x_batch.x.unsqueeze(1) 

                if args.model == 'PGCN':
                    x_batch = x_batch.x.view(-1,args.num_electrodes,5)
            if args.model == 'FC_STGNN' or args.model == 'FocalMoE' or args.model == 'moe':

                if isinstance(x_batch.x, list):
                    x_batch = torch.stack([torch.tensor(x, dtype=torch.float32) for x in x_batch.x])
                    if CUDA:
                        x_batch = x_batch.cuda()
                else:
                    x_batch = x_batch.x
                    if CUDA:
                        x_batch = x_batch.cuda()
                # print("Original x_batch shape:", x_batch.shape)
                batch_size = x_batch.size(0)
                total_size = x_batch.numel()
                # expected_size = batch_size * args.num_electrodes * 5
                # if total_size != expected_size:
                #     print(f"Warning: total_size ({total_size}) != expected_size ({expected_size})")
                x_batch = x_batch.view(-1,args.num_electrodes,args.feature_length)
            if args.model == 'HierCorrPool':
                x_batch = x_batch.x.view(-1,args.num_electrodes,args.feature_length)

            if args.model == 'SoftShapeNet':
                if isinstance(x_batch.x, list):
                    x_batch = torch.stack([torch.tensor(x, dtype=torch.float32) for x in x_batch.x])
                    if CUDA:
                        x_batch = x_batch.cuda()
                else:
                    x_batch = x_batch.x
                    if CUDA:
                        x_batch = x_batch.cuda()
                x_batch = x_batch.view(-1,args.num_electrodes,args.feature_length)

            x_batch = x_batch.to('cuda')
            try:
                out = net(x_batch)
            except Exception as e:
                if DiagEapatchDone is not None and isinstance(e, DiagEapatchDone):
                    print(f'[diag-eapatch] early stop predict: {e}')
                    break
                raise

            # LE probe side-channel (no compute change): accumulate A/B/C/D
            if do_probe and getattr(net, '_probe_buf', None) is not None:
                pb = net._probe_buf
                if pb.get('de_raw') is not None:
                    probe_acc['de_raw'].append(pb['de_raw'].numpy())
                if pb.get('H_b') is not None:
                    probe_acc['H_b'].append(pb['H_b'].numpy())
                if pb.get('attn_out') is not None:
                    probe_acc['attn_out'].append(pb['attn_out'].numpy())
                if pb.get('Z_b') is not None:
                    probe_acc['Z_b'].append(pb['Z_b'].numpy())
                if pb.get('de_patch') is not None:
                    probe_acc.setdefault('de_patch', []).append(pb['de_patch'].numpy())
                probe_acc['y'].append(y_batch.detach().cpu().numpy())

            if args.model == 'moe' and isinstance(out, tuple):
                logits, aux_loss = out
                main_loss = loss_fn(logits, y_batch)
                # loss = main_loss + args.loss_coef * aux_loss
                loss = main_loss
                _, pred = torch.max(logits, 1)
                probs = torch.softmax(logits, dim=1)[:, 1]
                logits_acc.append(logits.detach().cpu().numpy())
                if getattr(net, 'last_y_loc', None) is not None and getattr(net, 'last_y_glob', None) is not None:
                    le_logits_acc.append(net.last_y_loc.detach().cpu().numpy())
                    ge_logits_acc.append(net.last_y_glob.detach().cpu().numpy())
                if getattr(net, 'last_alpha', None) is not None:
                    a = net.last_alpha
                    if torch.is_tensor(a):
                        alpha_acc.append(a.detach().float().cpu().numpy().ravel())
            else:
                loss = loss_fn(out, y_batch)
                _, pred = torch.max(out, 1)
                probs = torch.softmax(out, dim=1)[:, 1]
                logits_acc.append(out.detach().cpu().numpy())
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
            score_val.extend(probs.cpu().tolist())

    # Persist LE probe dumps (train/test per fold) — side-channel only
    if do_probe and probe_acc is not None and probe_acc['y']:
        try:
            import os
            from pathlib import Path
            out_dir = Path(getattr(args, 'le_probe_dir', './save/le_probe'))
            out_dir.mkdir(parents=True, exist_ok=True)
            # Skip noisy val dumps during every epoch — only train/test tags
            if str(log_tag) in ('train', 'test', 'probe_train', 'probe_test'):
                tag = 'train' if 'train' in str(log_tag) else 'test'
                path = out_dir / f'sub{int(subject)}_fold{int(fold)}_{tag}.npz'
                payload = {'y': np.concatenate(probe_acc['y'], axis=0)}
                want = str(getattr(args, 'le_probe_keys', 'de_raw,H_b,attn_out,Z_b,de_patch'))
                want_set = {k.strip() for k in want.split(',') if k.strip()}
                for k in ('de_raw', 'H_b', 'attn_out', 'Z_b', 'de_patch'):
                    if k in want_set and probe_acc.get(k):
                        payload[k] = np.concatenate(probe_acc[k], axis=0)
                np.savez_compressed(path, **payload)
                print(f'[le-probe-dump] saved {path} keys={list(payload.keys())} '
                      f'n={payload["y"].shape[0]}')
        except Exception as e:
            print(f'[le-probe-dump] save failed: {e}')

    # Threshold-free / threshold-sensitive diagnostics (per call = per outer fold at test)
    auc = safe_roc_auc(act_val, score_val)
    macro_f1, bal_acc = get_diag_metrics(pred_val, act_val)
    pred_pos_rate = float(np.mean(np.asarray(pred_val) == 1)) if len(pred_val) else 0.0
    auc_s = f'{auc:.4f}' if auc == auc else 'nan'
    print(
        f'[predict-diag] tag={log_tag} sub={subject} fold={fold} '
        f'AUC={auc_s} pred_pos_rate={pred_pos_rate:.4f} '
        f'bal_ACC={bal_acc:.4f} macro_F1={macro_f1:.4f}'
    )
    branch = None
    if le_logits_acc and ge_logits_acc:
        branch = {
            'le': np.concatenate(le_logits_acc, axis=0),
            'ge': np.concatenate(ge_logits_acc, axis=0),
        }
    if logits_acc:
        if branch is None:
            branch = {}
        branch['logits'] = np.concatenate(logits_acc, axis=0)
    if alpha_acc:
        if branch is None:
            branch = {}
        branch['alpha'] = np.concatenate(alpha_acc, axis=0)
    return vl.item(), pred_val, act_val, score_val, branch

# def predict(data_loader, net, loss_fn):
#     net.eval()
#     pred_val = []
#     act_val = []
#     vl = Averager()
#     with torch.no_grad():
#         for i, (x_batch, y_batch) in enumerate(data_loader):
#             if CUDA:
#                 x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
#             out = net(x_batch)
#             loss = loss_fn(out, y_batch)
#             _, pred = torch.max(out, 1)
#             vl.add(loss.item())
#             pred_val.extend(pred.data.tolist())
#             act_val.extend(y_batch.data.tolist())
#     return vl.item(), pred_val, act_val

def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train(args, data_train, label_train, data_val, label_val, subject, fold):
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_fold' + str(fold)
    set_up(args)

    # P0 checkpoint hygiene: drop stale candidate from previous subject/fold
    candidate_path = osp.join(args.save_path, 'candidate.pth')
    if os.path.exists(candidate_path):
        os.remove(candidate_path)
        print(f'[ckpt] removed stale {candidate_path} before train sub={subject} fold={fold}')
    candidate_val = osp.join(args.save_path, 'candidate_val.npz')
    if os.path.exists(candidate_val):
        os.remove(candidate_val)

    train_loader = get_dataloader(args = args, data=data_train, label=label_train, batch_size = args.batch_size)

    val_loader = get_dataloader(args = args, data = data_val, label = label_val, batch_size = args.batch_size)

    model = get_model(args)
    if CUDA:
        model = model.cuda()

    # F3: α-head gets a smaller lr (structural; empty → single group)
    alpha_params = []
    if hasattr(model, 'fusion_alpha_parameters'):
        alpha_params = list(model.fusion_alpha_parameters())
    alpha_ids = {id(p) for p in alpha_params}
    base_params = [p for p in model.parameters() if id(p) not in alpha_ids]
    lr_main = float(args.learning_rate)
    lr_alpha = lr_main * float(getattr(args, 'fusion_alpha_lr_scale', 0.1))
    if alpha_params:
        optimizer = torch.optim.AdamW(
            [
                {'params': base_params, 'lr': lr_main},
                {'params': alpha_params, 'lr': lr_alpha},
            ],
            weight_decay=5e-3, betas=(0.9, 0.999), eps=1e-8,
        )
        print(f'[fusion-α] param_group: n_alpha={len(alpha_params)} '
              f'lr_main={lr_main:g} lr_alpha={lr_alpha:g}')
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr_main, weight_decay=5e-3, betas=(0.9, 0.999), eps=1e-8
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)    

    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        cw = getattr(args, 'class_weight', 'none')
        if cw == 'balanced':
            y_np = label_train.detach().cpu().numpy().astype(int).ravel()
            classes, counts = np.unique(y_np, return_counts=True)
            inv = counts.sum() / (len(classes) * counts.astype(np.float64))
            n_cls = int(getattr(args, 'num_class', int(classes.max()) + 1))
            weight = torch.ones(n_cls, dtype=torch.float32)
            for c, w in zip(classes, inv):
                weight[int(c)] = float(w)
            if CUDA:
                weight = weight.cuda()
            print(f'[class-weight] balanced weights={weight.tolist()} '
                  f'counts={dict(zip(classes.tolist(), counts.tolist()))}')
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()


    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    def save_candidate_val(y_true, y_score, branch=None):
        """Pair val probs (+ optional LE/GE/branch logits) with candidate.pth (never test)."""
        path = osp.join(args.save_path, 'candidate_val.npz')
        if os.path.exists(path):
            os.remove(path)
        payload = dict(
            y_true=np.asarray(y_true, dtype=np.int64),
            y_score=np.asarray(y_score, dtype=np.float64),
        )
        if branch is not None:
            if 'le' in branch and 'ge' in branch:
                payload['le_logits'] = np.asarray(branch['le'], dtype=np.float64)
                payload['ge_logits'] = np.asarray(branch['ge'], dtype=np.float64)
            if 'logits' in branch:
                payload['logits'] = np.asarray(branch['logits'], dtype=np.float64)
        np.savez_compressed(path, **payload)

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    # Init to -1 so first epoch always wins under strict > / >= (avoids never-saving when F1==0)
    trlog['max_acc'] = -1.0
    trlog['F1'] = 0.0
    trlog['max_F1'] = -1.0
    trlog['max_AUC'] = -1.0

    timer = Timer()
    ESC = args.early_stop_counter
    counter = 0

    for epoch in range(1, args.max_epoch + 1):

        if hasattr(model, 'set_fusion_epoch'):
            model.set_fusion_epoch(epoch)

        try:
            loss_train, pred_train, act_train, train_auc = train_one_epoch(
                args, data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer
            )
        except Exception as e:
            if DiagEapatchDone is not None and isinstance(e, DiagEapatchDone):
                print(f'[diag-eapatch] caught in train() after train_one_epoch: {e}')
                loss_train, pred_train, act_train, train_auc = 0.0, [], [], float('nan')
            else:
                raise

        if len(pred_train) == 0:
            acc_train, f1_train = 0.0, 0.0
        else:
            acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        auc_s = f'{train_auc:.4f}' if train_auc == train_auc else 'nan'
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f} AUC={} lr={:.8f}'
              .format(epoch, loss_train, acc_train, f1_train, auc_s, optimizer.param_groups[0]['lr']))

        try:
            loss_val, pred_val, act_val, score_val, branch_val = predict(
                args, data_loader=val_loader, net=model, loss_fn=loss_fn,
                subject=subject, fold=fold, log_tag='val'
            )
        except Exception as e:
            if DiagEapatchDone is not None and isinstance(e, DiagEapatchDone):
                print(f'[diag-eapatch] caught in train() after predict: {e}')
                raise
            raise
            branch_val = None  # unreachable; keep linter calm

        if str(getattr(args, 'diag_eapatch', 'off')).lower() == 'on':
            # After val pass, if enough samples collected, stop the whole run
            try:
                from moe import EnAsEmbedding
                if (EnAsEmbedding._diag_train_n >= EnAsEmbedding._DIAG_MAX_PER_MODE
                        and EnAsEmbedding._diag_eval_n >= EnAsEmbedding._DIAG_MAX_PER_MODE):
                    raise DiagEapatchDone(
                        f"diag-eapatch complete train={EnAsEmbedding._diag_train_n} "
                        f"eval={EnAsEmbedding._diag_eval_n}"
                    )
            except DiagEapatchDone:
                raise
            except Exception:
                pass

        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))

        # if acc_val >= trlog['max_acc']:
        #     trlog['max_acc'] = acc_val
        #     trlog['F1'] = f1_val
        #     save_model('candidate')
        #     counter = 0
        if ckpt_select_mode(args.load_path) == 'f1':
            # Monitor F1: strict > so a F1 plateau can early-stop (avoids burning max_epoch)
            if f1_val > trlog['max_F1']:
                trlog['max_acc'] = acc_val
                trlog['F1'] = f1_val
                trlog['max_F1'] = f1_val
                trlog['max_AUC'] = safe_roc_auc(act_val, score_val)
                save_model('candidate')
                save_candidate_val(act_val, score_val, branch=branch_val)
                counter = 0
                print(f'[select=f1] new best val_f1={f1_val:.4f} (acc={acc_val:.4f})')
            else:
                counter += 1
                if counter >= ESC:
                    print('==============Early Stopping==============')
                    break

        if ckpt_select_mode(args.load_path) == 'acc':
            # Keep original ACC protocol (>=) for baseline A/B
            if acc_val >= trlog['max_acc']:
                trlog['max_acc'] = acc_val
                trlog['F1'] = f1_val
                trlog['max_F1'] = f1_val
                trlog['max_AUC'] = safe_roc_auc(act_val, score_val)
                save_model('candidate')
                save_candidate_val(act_val, score_val, branch=branch_val)
                counter = 0
            else:
                counter += 1
                if counter >= ESC:
                    print('==============Early Stopping==============')
                    break

        if ckpt_select_mode(args.load_path) == 'auc':
            # Monitor val AUC (safe_roc_auc): strict >; init -1.0 so first valid epoch always wins
            auc_val = safe_roc_auc(act_val, score_val)
            auc_ok = auc_val == auc_val  # not NaN
            if auc_ok and auc_val > trlog['max_AUC']:
                trlog['max_acc'] = acc_val
                trlog['F1'] = f1_val
                trlog['max_F1'] = f1_val
                trlog['max_AUC'] = auc_val
                save_model('candidate')
                save_candidate_val(act_val, score_val, branch=branch_val)
                counter = 0
                print(f'[select=auc] new best val_auc={auc_val:.4f} (f1={f1_val:.4f} acc={acc_val:.4f})')
            else:
                counter += 1
                if counter >= ESC:
                    print('==============Early Stopping==============')
                    break

        # Collapse warning (majority predict): high ACC + near-zero F1
        if acc_val >= 0.70 and f1_val <= 1e-8:
            print(f'[WARN] possible majority collapse: val_acc={acc_val:.4f} val_f1={f1_val:.4f}')


        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)
        
        scheduler.step()

        print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject, fold))
    # save the training log file
    save_name = 'trlog' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = osp.join(args.save_path, experiment_setting, 'log_train')
    ensure_path(save_path)
    torch.save(trlog, osp.join(save_path, save_name))

    # Optional α_b dump (non-warmup batches only); default off → no file I/O
    if str(getattr(args, 'dump_alpha', 'off')).lower() == 'on':
        try:
            chunks = getattr(model, '_alpha_dump_chunks', None) or []
            if chunks:
                arr = np.concatenate(chunks, axis=0).astype(np.float64)
                a_min = float(getattr(args, 'fusion_alpha_min', 0.5))
                near = float(np.mean(arr <= (a_min + 0.02)))
                out_dir = osp.join(args.save_path, 'alpha_dump')
                ensure_path(out_dir)
                tag = getattr(args, 'run_tag', '') or 'untagged'
                out_p = osp.join(out_dir, f'{tag}_sub{int(subject)}_fold{int(fold)}.npz')
                np.savez_compressed(
                    out_p,
                    alpha=arr,
                    mean=float(arr.mean()),
                    std=float(arr.std()),
                    median_batch_proxy_std=float(arr.std()),  # sample-level; log has batch std
                    frac_near_alpha_min=near,
                    alpha_min=a_min,
                    n=int(arr.size),
                )
                print(f'[dump-alpha] saved {out_p} n={arr.size} mean={arr.mean():.4f} '
                      f'std={arr.std():.4f} frac_near_min={near:.3f}')
            if hasattr(model, '_alpha_dump_chunks'):
                model._alpha_dump_chunks = []
        except Exception as e:
            print(f'[dump-alpha] failed: {e}')

    # GE routing diagnostics (per fold → process-level accum)
    if str(getattr(args, 'ge_diag_routing', 'off')).lower() == 'on':
        try:
            ge = getattr(model, 'GMOE', None)
            if ge is not None and hasattr(ge, 'routing_reports'):
                for r in ge.routing_reports():
                    print(
                        f"[GE-routing] sub={subject} fold={fold} layer={r.get('layer')} "
                        f"I(v;e)={r['I_ve']:.6f} "
                        f"I_shuf={r['I_shuffle_mean']:.6f}±{r['I_shuffle_std']:.6f} "
                        f"gamma_max_mean={r['gamma_max_mean']:.4f} "
                        f"expert_frac={[round(x,3) for x in r['expert_frac']]} "
                        f"‖ΔW_g‖={r['W_g_delta_norm']:.4f} (init‖W_g‖={r['W_g_init_norm']:.4f}) "
                        f"n={int(r['n_tokens'])}"
                    )
                    print(f"[GE-routing] counts=\n{r['counts']}")
                for layer in getattr(ge, 'layers', []):
                    if getattr(layer, 'gcn', None) is not None:
                        layer.gcn.flush_routing_to_global()
        except Exception as e:
            print(f'[GE-routing] report failed: {e}')

    return trlog['max_acc'], trlog['F1'], trlog['max_F1'], trlog['max_AUC']


def test(args, data, label, reproduce, subject, fold):
    set_up(args)
    seed_all(args.random_seed)
    batch_size = args.batch_size
    # Deterministic order for paired bootstrap dumps across configs.
    test_loader = get_dataloader(
        args=args, data=data, label=label, batch_size=batch_size, shuffle=False
    )

    model = get_model(args)
    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    if reproduce:
        model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
        data_type = 'model_{}_{}'.format(args.data_using, args.label_type)
        experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
        load_path_final = osp.join(args.save_path, experiment_setting, data_type, model_name_reproduce)
        model.load_state_dict(torch.load(load_path_final))
    else:
        model.load_state_dict(torch.load(args.load_path_final))


    loss, pred, act, score, branch_te = predict(
        args, data_loader=test_loader, net=model, loss_fn=loss_fn,
        subject=subject, fold=fold, log_tag='test'
    )
    # Dual report: t=0.5 (argmax) vs val-selected t*
    thr_obj = str(getattr(args, 'thr_objective', 'macro_f1'))
    pred_05 = list(pred)
    acc_05, f1_05, cm = get_metrics(y_pred=pred_05, y_true=act)
    fold_info = fold_collapse_report(pred_05, act, subject=subject, fold=fold, y_score=score)
    fold_info['y_score'] = list(score)
    fold_info['y_true'] = list(act)
    fold_info['thr_05'] = 0.5
    fold_info['acc_05'] = acc_05
    fold_info['f1_05'] = f1_05
    print('>>> Test@0.5:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc_05, f1_05))

    # Persist per-fold val+test logits for offline frozen fusion (B2); never used to pick α here
    try:
        tag = getattr(args, 'run_tag', '') or 'untagged'
        fold_dir = osp.join(args.save_path, 'pred_scores', 'folds')
        ensure_path(fold_dir)
        val_npz = osp.join(args.save_path, 'max-f1_val.npz')
        if ckpt_select_mode(args.load_path) == 'acc':
            val_npz = osp.join(args.save_path, 'max-acc_val.npz')
        elif ckpt_select_mode(args.load_path) == 'auc':
            val_npz = osp.join(args.save_path, 'max-auc_val.npz')
        payload = {
            'subject': int(subject),
            'fold': int(fold),
            'y_true_test': np.asarray(act, dtype=np.int64),
            'y_score_test': np.asarray(score, dtype=np.float64),
        }
        if branch_te is not None and 'logits' in branch_te:
            payload['logits_test'] = np.asarray(branch_te['logits'], dtype=np.float64)
        if branch_te is not None and 'le' in branch_te and 'ge' in branch_te:
            payload['le_logits_test'] = np.asarray(branch_te['le'], dtype=np.float64)
            payload['ge_logits_test'] = np.asarray(branch_te['ge'], dtype=np.float64)
        if branch_te is not None and 'alpha' in branch_te:
            payload['alpha_test'] = np.asarray(branch_te['alpha'], dtype=np.float64).ravel()
        if osp.exists(val_npz):
            vd = np.load(val_npz)
            payload['y_true_val'] = np.asarray(vd['y_true'], dtype=np.int64)
            payload['y_score_val'] = np.asarray(vd['y_score'], dtype=np.float64)
            if 'logits' in vd.files:
                payload['logits_val'] = np.asarray(vd['logits'], dtype=np.float64)
            if 'le_logits' in vd.files and 'ge_logits' in vd.files:
                payload['le_logits_val'] = np.asarray(vd['le_logits'], dtype=np.float64)
                payload['ge_logits_val'] = np.asarray(vd['ge_logits'], dtype=np.float64)
            if 'alpha' in vd.files:
                payload['alpha_val'] = np.asarray(vd['alpha'], dtype=np.float64)
        out_fold = osp.join(fold_dir, f'{tag}_sub{int(subject)}_fold{int(fold)}.npz')
        np.savez_compressed(out_fold, **payload)
        print(f'[fold-dump] saved {out_fold} keys={sorted(payload.keys())}')
    except Exception as e:
        print(f'[fold-dump] failed: {e}')

    # ---- Module B: val_select α* (never use test to pick α) ----
    alpha_mode = str(getattr(args, 'alpha_mode', 'fixed')).lower()
    fold_info['alpha_mode'] = alpha_mode
    fold_info['alpha_train'] = float(getattr(args, 'fusion_alpha', 0.9))
    fold_info['alpha_star'] = None
    fold_info['auc_a05'] = None
    fold_info['auc_astar'] = None
    if (
        alpha_mode == 'val_select'
        and str(getattr(args, 'branch', '')).lower() == 'le_ge'
        and branch_te is not None
    ):
        val_npz = osp.join(args.save_path, 'max-f1_val.npz')
        if ckpt_select_mode(args.load_path) == 'acc':
            val_npz = osp.join(args.save_path, 'max-acc_val.npz')
        elif ckpt_select_mode(args.load_path) == 'auc':
            val_npz = osp.join(args.save_path, 'max-auc_val.npz')
        if osp.exists(val_npz):
            from utils import select_alpha_on_val, safe_roc_auc as _auc
            vd = np.load(val_npz)
            if 'le_logits' in vd.files and 'ge_logits' in vd.files:
                a_star, vm = select_alpha_on_val(
                    vd['y_true'], vd['le_logits'], vd['ge_logits'],
                    objective=str(getattr(args, 'alpha_objective', 'auc')),
                )
                # Fixed α=0.5 reference on TEST branch logits
                def _fuse_metrics(a, le, ge, yt):
                    fused = a * le + (1.0 - a) * ge
                    m = fused.max(axis=1, keepdims=True)
                    e = np.exp(fused - m)
                    prob = e / e.sum(axis=1, keepdims=True)
                    sc = prob[:, 1]
                    pr = (sc >= 0.5).astype(int).tolist()
                    ac, f1v, _ = get_metrics(y_pred=pr, y_true=yt)
                    return ac, f1v, float(_auc(yt, sc)), sc, pr

                le_te, ge_te = branch_te['le'], branch_te['ge']
                acc05, f105, auc05, sc05, pr05 = _fuse_metrics(0.5, le_te, ge_te, act)
                acc_s, f1_s, auc_s, sc_s, pr_s = _fuse_metrics(a_star, le_te, ge_te, act)
                print(
                    f'[alpha] val_select α*={a_star:.2f} (obj={getattr(args,"alpha_objective","auc")}) '
                    f'val_obj={vm.get("objective_score", float("nan")):.4f} | '
                    f'test@α0.5 AUC={auc05:.4f} ACC={acc05:.4f} F1={f105:.4f} | '
                    f'test@α* AUC={auc_s:.4f} ACC={acc_s:.4f} F1={f1_s:.4f}'
                )
                fold_info['alpha_star'] = a_star
                fold_info['auc_a05'] = auc05
                fold_info['acc_a05'] = acc05
                fold_info['f1_a05'] = f105
                fold_info['auc_astar'] = auc_s
                fold_info['acc_astar'] = acc_s
                fold_info['f1_astar'] = f1_s
                # Primary reported metrics use α* (B1 protocol)
                acc, f1, pred_final = acc_s, f1_s, pr_s
                fold_info['y_score'] = list(sc_s)
                # refresh collapse under α*
                fold_info.update({
                    k: v for k, v in fold_collapse_report(
                        pr_s, act, subject=subject, fold=fold, y_score=sc_s
                    ).items() if k.startswith('collapse') or k in ('macro_f1', 'bal_acc', 'auc')
                })
            else:
                print(f'[alpha] {val_npz} missing le/ge logits — fall back to train-α fuse')
                acc, f1, pred_final = acc_05, f1_05, pred_05
        else:
            print(f'[alpha] {val_npz} missing — fall back to train-α fuse')
            acc, f1, pred_final = acc_05, f1_05, pred_05
    else:
        acc, f1, pred_final = acc_05, f1_05, pred_05

    t_star = 0.5
    if thr_obj != 'argmax' and alpha_mode != 'val_select':
        val_npz = osp.join(args.save_path, 'max-f1_val.npz')
        if ckpt_select_mode(args.load_path) == 'acc':
            val_npz = osp.join(args.save_path, 'max-acc_val.npz')
        elif ckpt_select_mode(args.load_path) == 'auc':
            val_npz = osp.join(args.save_path, 'max-auc_val.npz')
        if osp.exists(val_npz):
            from utils import select_threshold_on_val
            vd = np.load(val_npz)
            t_star, vm = select_threshold_on_val(vd['y_true'], vd['y_score'], objective=thr_obj)
            pred_star = (np.asarray(score) >= t_star).astype(int).tolist()
            acc_star, f1_star, _ = get_metrics(y_pred=pred_star, y_true=act)
            fold_star = fold_collapse_report(
                pred_star, act, subject=subject, fold=fold, y_score=score
            )
            print(
                f'>>> Test@t*: t*={t_star:.2f} obj={thr_obj} '
                f'acc={acc_star:.4f} f1={f1_star:.4f} '
                f'(val_obj={vm.get("objective_score", float("nan")) if vm else float("nan"):.4f})'
            )
            fold_info['thr_star'] = t_star
            fold_info['acc_star'] = acc_star
            fold_info['f1_star'] = f1_star
            fold_info['collapse_pos_star'] = fold_star.get('collapse_pos')
            fold_info['collapse_neg_star'] = fold_star.get('collapse_neg')
            # Report val-selected t* ACC/F1; AUC stays threshold-free on y_score
            acc, f1, pred_final = acc_star, f1_star, pred_star
            for _k in ('collapse_pos', 'collapse_neg', 'macro_f1', 'bal_acc'):
                if _k in fold_star:
                    fold_info[_k] = fold_star[_k]
        else:
            print(f'[thr] {val_npz} missing — report t=0.5 only')
            fold_info['thr_star'] = None
    else:
        fold_info['thr_star'] = 0.5 if alpha_mode != 'val_select' else None

    return acc, pred_final, act, fold_info


def combine_train(args, data, label, subject, fold, target_acc):
    save_name = '_sub' + str(subject) + '_fold' + str(fold)
    set_up(args)
    seed_all(args.random_seed)
    train_loader = get_dataloader(data, label, args.batch_size)
    model = get_model(args)
    if CUDA:
        model = model.cuda()
    model.load_state_dict(torch.load(args.load_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate*1e-1)

    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch_cmb + 1):
        loss, pred, act, _train_auc = train_one_epoch(
            args, data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer
        )
        acc, f1, _ = get_metrics(y_pred=pred, y_true=act)
        print('Stage 2 : epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss, acc, f1))

        if acc >= target_acc or epoch == args.max_epoch_cmb:
            print('early stopping!')
            save_model('final_model')
            # save model here for reproduce
            model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
            data_type = 'model_{}_{}'.format(args.data_using, args.label_type)
            experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
            save_path = osp.join(args.save_path, experiment_setting, data_type)
            ensure_path(save_path)
            model_name_reproduce = osp.join(save_path, model_name_reproduce)
            torch.save(model.state_dict(), model_name_reproduce)
            break

        trlog['train_loss'].append(loss)
        trlog['train_acc'].append(acc)

        print('ETA:{}/{} SUB:{} TRIAL:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject, fold))

    save_name = 'trlog_comb' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = osp.join(args.save_path, experiment_setting, 'log_train_cmb')
    ensure_path(save_path)
    torch.save(trlog, osp.join(save_path, save_name))
