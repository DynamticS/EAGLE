import os
import sys
import time
import h5py
import numpy as np
import pprint
import random
import torch
import torch.fft as fft
# from networks import *
# from eeg_dataset import *
# from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score,
)
import torch.nn as nn

from scipy.sparse import coo_matrix
from torch_geometric.data import Data, DataLoader
# from torcheeg.models import EEGNet, DGCNN, ArjunViT, STNet, TSCeption
# from torcheeg.models.pyg import RGNN
# from RGNN import SymSimGCNNet
# from torcheeg.transforms.pyg import ToG
# from FC_STGNN.Model import FC_STGNN
# from HierCorrPool.Model import HierCorrPool
# from SoftShapeModel import SoftShapeNet

def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(x)
    # devices = torch.device("cuda:%s" % (x) if torch.cuda.is_available() else "cpu")
    print('using gpu:', x)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def ckpt_select_mode(load_path):
    """Select inner-fold metric from checkpoint filename (max-f1 / max-acc / max-auc)."""
    base = os.path.basename(str(load_path or ''))
    if base == 'max-acc.pth':
        return 'acc'
    if base == 'max-auc.pth':
        return 'auc'
    return 'f1'


def get_model(args):
    if str(getattr(args, 'model', 'moe')).lower() != 'moe':
        raise ValueError('This release only supports --model moe')
    from moe import MoE
    if args.data_using == 'DEAP':
        input_size = 512
    else:
        input_size = int(getattr(args, 'feature_length', 512) or 512)
    return MoE(
        args,
        input_size,
        int(getattr(args, 'num_class', 2) or 2),
        args.num_experts,
        args.hidden_size,
        k=args.k,
        noisy_gating=True,
        num_node=32,
        patch_size=args.patch_size,
        num_patch=args.num_patch,
        feature_embed_dim=args.feature_embed_dim,
    )


# def get_dataloader(data, label, batch_size):
#     # load the data
#     dataset = eegDataset(data, label)
#     loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
#     return loader

def get_dataloader(args, data, label, batch_size, shuffle=True):
    # load the data
    # dataset = [Data(x=data[i], edge_index=get_edge_index(args), y=label[i]) for i in range(data.shape[0])]
    dataset = [Data(x=data[i], edge_index=get_edge_index(args), y=label[i]) for i in range(data.shape[0])]
    print("get_data")
    # Test/eval must use shuffle=False so pred dumps stay sample-aligned across
    # configs (trial vs patch MoE init consumes different RNG before shuffle).
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    return loader

def get_edge_index(args):

    if args.dense == 'full' and args.data_using == 'DEAP':
        nodes_num = 32
        edge_data = torch.ones([nodes_num, nodes_num])  # initialize edge index
        edge_index = coo_matrix(edge_data)
        edge_index = np.vstack((edge_index.row, edge_index.col))
        # edge_index = torch.from_numpy(edge_index).to(torch.int64).to(device)
        edge_index = torch.from_numpy(edge_index).to(torch.int64)

    else:
        edge_index = None

    return edge_index

def get_edge_attr(type, signal_patch, batch):
    CUDA = torch.cuda.is_available()
    if type == 'COS':
        # select one sample [0,1023]
        # select = torch.arange(0,(edge_index.shape[-1] // batch),1)
        # index = torch.index_select(edge_index, dim = 1, index = select.cuda())
        #select all
        # signal_patch = signal_patch.reshape(batch,signal_patch.shape[0]//batch,signal_patch.shape[-1])
        signal_patch = signal_patch.reshape(batch, signal_patch.shape[0] // batch,
                                            signal_patch.shape[-1])
        norms = torch.norm(signal_patch, dim=2, keepdim=True)
        signal_patch_normalized = signal_patch / norms
        sim_matrices = torch.bmm(signal_patch, signal_patch_normalized.transpose(1,2))
        edge_attr = sim_matrices.view(-1)
        edge_attr = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min())

    if type == 'PLI':

        signal_patch = signal_patch.reshape(batch, signal_patch.shape[0] // batch,
                                            signal_patch.shape[-1])

        # Calculate phase difference matrix
        phase_diff_matrices = calculate_phase_difference_matrix(signal_patch)

        # Normalize the phase difference matrices using sigmoid
        normalized_matrices = sigmoid_normalize_phase_diff_matrix(phase_diff_matrices)
        edge_attr = normalized_matrices.view(-1)


    return edge_attr


def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def get_diag_metrics(y_pred, y_true):
    """Diagnostic-only metrics (do NOT use for model selection)."""
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return float(macro_f1), float(bal_acc)


def safe_roc_auc(y_true, y_score, positive_label=1):
    """roc_auc_score with single-class / empty guard → float('nan')."""
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return float('nan')
    try:
        return float(roc_auc_score(yt, ys))
    except Exception:
        return float('nan')


def bootstrap_auc_ci(y_true, y_score, n_boot=1000, alpha=0.05, rng=None):
    """
    Bootstrap CI for a single subject's AUC (resample test predictions).
    Returns (point_auc, ci_lo, ci_hi) or nans if undefined.
    """
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    point = safe_roc_auc(yt, ys)
    if point != point or len(yt) < 2:
        return point, float('nan'), float('nan')
    rng = np.random.default_rng(rng)
    n = len(yt)
    boots = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        a = safe_roc_auc(yt[idx], ys[idx])
        if a == a:
            boots.append(a)
    if not boots:
        return point, float('nan'), float('nan')
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def bootstrap_mean_auc_ci(subject_y_true_list, subject_y_score_list,
                          n_boot=1000, alpha=0.05, seed=8989):
    """
    Estimator = mean over subjects of per-subject ROC-AUC (on that subject's
    pooled test scores). Bootstrap: within each subject, resample test indices
    with replacement, recompute subject AUC, then average across subjects;
    repeat n_boot times; report 2.5/97.5 percentiles of the subject-mean.

    Do NOT pool samples across subjects (Simpson bias when pos-rates differ).
    Returns dict: mean_auc, ci_lo, ci_hi, per_subject_aucs, se.
    """
    rng = np.random.default_rng(seed)
    n_sub = len(subject_y_true_list)
    if n_sub == 0:
        return {
            'mean_auc': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan'),
            'per_subject_aucs': [], 'se': float('nan'),
        }
    point_aucs = []
    for yt, ys in zip(subject_y_true_list, subject_y_score_list):
        point_aucs.append(safe_roc_auc(yt, ys))
    finite_pts = [a for a in point_aucs if a == a]
    mean_auc = float(np.mean(finite_pts)) if finite_pts else float('nan')

    means = []
    for _ in range(int(n_boot)):
        sub_aucs = []
        for yt, ys in zip(subject_y_true_list, subject_y_score_list):
            yt = np.asarray(yt).ravel()
            ys = np.asarray(ys, dtype=np.float64).ravel()
            if len(yt) == 0:
                continue
            idx = rng.integers(0, len(yt), size=len(yt))
            a = safe_roc_auc(yt[idx], ys[idx])
            if a == a:
                sub_aucs.append(a)
        if sub_aucs:
            means.append(float(np.mean(sub_aucs)))
    if means:
        ci_lo = float(np.percentile(means, 100 * (alpha / 2)))
        ci_hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
        se = float(np.std(means, ddof=1)) if len(means) > 1 else float('nan')
    else:
        ci_lo = ci_hi = se = float('nan')
    return {
        'mean_auc': mean_auc,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'per_subject_aucs': point_aucs,
        'se': se,
    }


def paired_bootstrap_delta_auc(
    subject_y_true_list,
    scores_a_list,
    scores_b_list,
    n_boot=1000,
    alpha=0.05,
    seed=8989,
):
    """
    Paired bootstrap ΔAUC = mean_i(AUC_a,i - AUC_b,i).
    For each bootstrap replicate, resample the SAME indices for both configs
    (shared test set under KFold=8989), compute per-subject ΔAUC, then mean.
    Returns dict with delta_mean, ci_lo, ci_hi, se, per_subject_delta.
    """
    rng = np.random.default_rng(seed)
    n_sub = len(subject_y_true_list)
    if n_sub == 0 or n_sub != len(scores_a_list) or n_sub != len(scores_b_list):
        return {
            'delta_mean': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan'),
            'se': float('nan'), 'per_subject_delta': [],
        }
    point_deltas = []
    for yt, sa, sb in zip(subject_y_true_list, scores_a_list, scores_b_list):
        yt = np.asarray(yt).ravel()
        sa = np.asarray(sa, dtype=np.float64).ravel()
        sb = np.asarray(sb, dtype=np.float64).ravel()
        n = min(len(yt), len(sa), len(sb))
        yt, sa, sb = yt[:n], sa[:n], sb[:n]
        aa, bb = safe_roc_auc(yt, sa), safe_roc_auc(yt, sb)
        point_deltas.append((aa - bb) if (aa == aa and bb == bb) else float('nan'))
    finite_pts = [d for d in point_deltas if d == d]
    delta_mean = float(np.mean(finite_pts)) if finite_pts else float('nan')

    boots = []
    for _ in range(int(n_boot)):
        sub_d = []
        for yt, sa, sb in zip(subject_y_true_list, scores_a_list, scores_b_list):
            yt = np.asarray(yt).ravel()
            sa = np.asarray(sa, dtype=np.float64).ravel()
            sb = np.asarray(sb, dtype=np.float64).ravel()
            n = min(len(yt), len(sa), len(sb))
            if n < 2:
                continue
            yt, sa, sb = yt[:n], sa[:n], sb[:n]
            idx = rng.integers(0, n, size=n)
            aa = safe_roc_auc(yt[idx], sa[idx])
            bb = safe_roc_auc(yt[idx], sb[idx])
            if aa == aa and bb == bb:
                sub_d.append(aa - bb)
        if sub_d:
            boots.append(float(np.mean(sub_d)))
    if boots:
        ci_lo = float(np.percentile(boots, 100 * (alpha / 2)))
        ci_hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
        se = float(np.std(boots, ddof=1)) if len(boots) > 1 else float('nan')
    else:
        ci_lo = ci_hi = se = float('nan')
    return {
        'delta_mean': delta_mean,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'se': se,
        'per_subject_delta': point_deltas,
    }
def fold_collapse_report(y_pred, y_true, subject=None, fold=None, positive_label=1, y_score=None):
    """
    Per-fold collapse diagnostics. Selection metric remains binary F1 elsewhere.
    Returns dict with rates, CM cells, flags, and threshold-free AUC when y_score given.
    """
    import numpy as np
    yp = np.asarray(y_pred).astype(int).ravel()
    yt = np.asarray(y_true).astype(int).ravel()
    n = max(len(yp), 1)
    pred_pos_rate = float((yp == positive_label).mean()) if len(yp) else 0.0
    true_pos_rate = float((yt == positive_label).mean()) if len(yt) else 0.0
    # confusion: rows true, cols pred for labels [0,1]
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    macro_f1, bal_acc = get_diag_metrics(yp, yt)
    auc = float('nan')
    if y_score is not None:
        auc = safe_roc_auc(yt, y_score, positive_label=positive_label)
    tag = ''
    if pred_pos_rate >= 1.0 - 1e-12:
        tag = '[COLLAPSE-POS]'
    elif pred_pos_rate <= 1e-12:
        tag = '[COLLAPSE-NEG]'
    auc_s = f'{auc:.4f}' if auc == auc else 'nan'
    print(
        f'[fold-diag] sub={subject} fold={fold} {tag} '
        f'pred_pos_rate={pred_pos_rate:.4f} true_pos_rate={true_pos_rate:.4f} '
        f'TP={tp} FP={fp} TN={tn} FN={fn} macro_F1={macro_f1:.4f} bal_ACC={bal_acc:.4f} '
        f'AUC={auc_s}'
    )
    return {
        'pred_pos_rate': pred_pos_rate,
        'true_pos_rate': true_pos_rate,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'macro_f1': macro_f1,
        'bal_acc': bal_acc,
        'auc': auc,
        'collapse_pos': tag == '[COLLAPSE-POS]',
        'collapse_neg': tag == '[COLLAPSE-NEG]',
        'tag': tag,
    }


def select_threshold_on_val(y_true, y_score, objective='macro_f1',
                            grid=None):
    """Pick decision threshold on validation probs ONLY (never touch test).

    objective:
      argmax       → fixed 0.5
      macro_f1     → maximize macro-F1 (default; avoids pure-F1 majority bias)
      acc_f1_mean  → maximize 0.5*(ACC+binary F1)
      balanced_acc → maximize balanced accuracy
    Returns (t_star, metrics_at_t_star).
    """
    yt = np.asarray(y_true).astype(int).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    if objective == 'argmax' or len(yt) == 0:
        t_star = 0.5
        pred = (ys >= t_star).astype(int)
        acc, f1, _ = get_metrics(y_pred=pred, y_true=yt) if len(yt) else (float('nan'), float('nan'), None)
        mf1, bacc = (get_diag_metrics(pred, yt) if len(yt) else (float('nan'), float('nan')))
        return t_star, dict(acc=acc, f1=f1, macro_f1=mf1, balanced_acc=bacc)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    best_t, best_score, best_m = 0.5, -1.0, None
    for t in grid:
        pred = (ys >= float(t)).astype(int)
        acc, f1, _ = get_metrics(y_pred=pred, y_true=yt)
        mf1, bacc = get_diag_metrics(pred, yt)
        if objective == 'macro_f1':
            score = mf1
        elif objective == 'acc_f1_mean':
            score = 0.5 * (acc + f1)
        elif objective == 'balanced_acc':
            score = bacc
        else:
            raise ValueError(f'unknown thr objective {objective}')
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_m = dict(acc=acc, f1=f1, macro_f1=mf1, balanced_acc=bacc, objective_score=score)
    return best_t, best_m


def select_alpha_on_val(y_true, le_logits, ge_logits, objective='auc', n_grid=101):
    """Pick α* on VALIDATION branch logits only (never test).

    Fuse: y = α·LE + (1−α)·GE in logit space, then score = softmax(y)[:,1].
    le_logits / ge_logits: [N, K] or [N] (pos-class logit / margin).
    Returns (alpha_star, metrics_dict).
    """
    yt = np.asarray(y_true).astype(int).ravel()
    le = np.asarray(le_logits, dtype=np.float64)
    ge = np.asarray(ge_logits, dtype=np.float64)
    if le.ndim == 1:
        # reconstruct 2-class logits from pos score via logit if needed — prefer [N,2]
        raise ValueError('select_alpha_on_val expects le/ge logits with shape [N, K]')
    alphas = np.linspace(0.0, 1.0, int(n_grid))
    best_a, best_score, best_m = 0.5, -1.0, {}
    for a in alphas:
        fused = a * le + (1.0 - a) * ge
        # softmax pos class
        m = fused.max(axis=1, keepdims=True)
        e = np.exp(fused - m)
        prob = e / e.sum(axis=1, keepdims=True)
        score = prob[:, 1]
        pred = (score >= 0.5).astype(int)
        acc, f1, _ = get_metrics(y_pred=pred, y_true=yt)
        mf1, _ = get_diag_metrics(pred, yt)
        auc = safe_roc_auc(yt, score)
        if objective == 'auc':
            obj = auc if auc == auc else -1.0
        elif objective == 'macro_f1':
            obj = mf1
        elif objective == 'acc_f1_mean':
            obj = 0.5 * (acc + f1)
        else:
            raise ValueError(f'unknown alpha objective {objective}')
        if obj > best_score:
            best_score = float(obj)
            best_a = float(a)
            best_m = dict(acc=acc, f1=f1, macro_f1=mf1, auc=auc, objective_score=obj)
    return best_a, best_m


def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def L2Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(w.pow(2))
    return err


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def hilbert_torch(x):
    """
    Compute the analytic signal using Hilbert transform.
    x: input tensor of shape (..., n_samples)
    """
    N = x.shape[-1]
    Xf = fft.fft(x, dim=-1)
    h = torch.zeros(N, dtype=torch.complex64, device=x.device)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return fft.ifft(Xf * h, dim=-1)

def calculate_phase_difference_matrix(signals):
    """
    Calculate the phase difference matrix for multiple signals using vectorized operations.
    signals: tensor of shape (batch_size, n_signals, n_samples)
    """
    # Calculate analytic signals for all inputs
    analytic_signals = hilbert_torch(signals)
    
    # Calculate phases for all signals
    phases = torch.angle(analytic_signals)  # shape: (batch_size, n_signals, n_samples)
    
    # Calculate all pairwise phase differences
    phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)  # shape: (batch_size, n_signals, n_signals, n_samples)
    
    # Wrap phase difference to [-pi, pi]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    
    # Calculate mean phase difference
    phase_diff_matrix = phase_diff.mean(dim=-1)  # shape: (batch_size, n_signals, n_signals)
    
    return phase_diff_matrix

def sigmoid_normalize_phase_diff_matrix(phase_diff_matrix):
    """
    Normalize the phase difference matrix using sigmoid function.
    """
    return torch.sigmoid(phase_diff_matrix)


