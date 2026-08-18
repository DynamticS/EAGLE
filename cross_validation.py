import numpy as np
import datetime
import os
import h5py
import copy
import os.path as osp
from train_model_eeg import *
from utils import Averager, ensure_path, get_metrics, get_diag_metrics, bootstrap_mean_auc_ci, safe_roc_auc, ckpt_select_mode
from sklearn.model_selection import KFold
import time

# DEAP: 40 trials × 15 segments = 600. Labels are trial-level.
SEG_PER_TRIAL_DEAP = 15
N_TRIALS_DEAP = 40
KFOLD_RANDOM_STATE = 8989


def expand_trial_indices(trial_idx, seg_per_trial=SEG_PER_TRIAL_DEAP):
    """Map trial indices → segment indices (contiguous blocks)."""
    segs = []
    for t in np.asarray(trial_idx, dtype=np.int64).ravel():
        t = int(t)
        segs.extend(range(t * seg_per_trial, (t + 1) * seg_per_trial))
    return np.asarray(segs, dtype=np.int64)


def iter_outer_splits(n_segments, fold, split_level, seg_per_trial=SEG_PER_TRIAL_DEAP,
                      random_state=KFOLD_RANDOM_STATE):
    """Yield (idx_train, idx_test) over segments. trial mode: no trial crosses the split."""
    if split_level == 'segment':
        kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)
        yield from kf.split(np.arange(n_segments))
        return
    if split_level != 'trial':
        raise ValueError(f'unknown split_level={split_level!r}')
    if n_segments % seg_per_trial != 0:
        raise ValueError(f'n_segments={n_segments} not divisible by seg_per_trial={seg_per_trial}')
    n_trials = n_segments // seg_per_trial
    kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)
    for tr_t, te_t in kf.split(np.arange(n_trials)):
        assert len(set(tr_t) & set(te_t)) == 0, 'trial split leaked trial ids'
        idx_train = expand_trial_indices(tr_t, seg_per_trial)
        idx_test = expand_trial_indices(te_t, seg_per_trial)
        assert len(set(idx_train) & set(idx_test)) == 0
        yield idx_train, idx_test

ROOT = os.getcwd()


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = args.model
        self.data_dir = args.data_path
        self.label_type = args.label_type
        self.data_using = args.data_using
        self.cts = args.cts
        self.data_norm = args.data_norm
        self.load_path = args.load_path
        # Log the results per subject
        result_path = osp.join(args.save_path, 'result')
        ensure_path(result_path)
        self.text_file = osp.join(result_path,
                                  "results_{}.txt".format(args.data_using))
        file = open(self.text_file, 'a')
        # file.write("\n" + str(datetime.datetime.now()) +
        #            "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
        #            "\n1)number_class:" + str(args.num_class) +
        #            "\n2)random_seed:" + str(args.random_seed) +
        #            "\n3)learning_rate:" + str(args.learning_rate) +
        #            "\n4)pool:" + str(args.pool) +
        #            "\n5)num_epochs:" + str(args.max_epoch) +
        #            "\n6)batch_size:" + str(args.batch_size) +
        #            "\n8)hidden_node:" + str(args.hidden) +
        #            "\n9)input_shape:" + str(args.input_shape) +
        #            "\n10)class:" + str(args.label_type) +
        #            "\n11)T:" + str(args.T) +
        #            "\n12)graph-type:" + str(args.graph_type) + '\n')
        # file.write("\n" + str(datetime.datetime.now()) +
        #            "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.data_using) +
        #            "\n0)Notice:" + str(args.notice) +
        #            "\n1)Label_type:" + str(args.label_type) +
        #            "\n2)random_seed:" + str(args.random_seed) +
        #            "\n3)learning_rate:" + str(args.learning_rate) +
        #            "\n4)esc:" + str(args.early_stop_counter) +
        #            "\n5)num_epochs:" + str(args.max_epoch) +
        #            "\n6)batch_size:" + str(args.batch_size) +
        #            "\n7)dropout:" + str(args.dropout) +
        #            "\n8)hidden_node:" + str(args.hidden) +
        #            "\n9)dense method:" + str(args.dense) +
        #            "\n10)max_epoch_combine_train:" + str(args.max_epoch_cmb) +
        #            "\n11)T:" + str(args.T) + 
        #            "\n12)T:" + str(args.edge_compute) +
        #            "\n13)combine train switch:" + str(args.cts) +
        #            "\n# moe parameters" +
        #            "\n14)num_experts:" + str(args.num_experts) +
        #            "\n15)k:" + str(args.k) +
        #            "\n16)hidden_size:" + str(args.hidden_size) +
        #            "\n17)loss_coef:" + str(args.loss_coef) +
        #            "\n# EnAs Embedding parameters" +
        #            "\n18)feature_embed_dim:" + str(args.feature_embed_dim) +
        #            "\n19)patch_size:" + str(args.patch_size) +
        #            "\n20)num_patch:" + str(args.num_patch) +
        #            "\n21)dropout_rate:" + str(args.dropout_rate) + '\n')

        try:
            args_dict = vars(args)
        except Exception:
            args_dict = args.__dict__ if hasattr(args, '__dict__') else str(args)
        print('[Args Dump]', args_dict)
        file.write("\n" + str(datetime.datetime.now()) + "\nArgs Dump:\n" + str(args_dict) + "\n")
        file.close()

    def load_per_subject(self, sub):
        """
        load data for sub
        :param sub: which subject's data to load
        :return: data and label
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = osp.join(save_path, data_type, sub_code)
        dataset = h5py.File(path, 'r')
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
        print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label

    # def prepare_data(self, idx_train, idx_test, data, label):
    #     """
    #     1. get training and testing data according to the index
    #     2. numpy.array-->torch.tensor
    #     :param idx_train: index of training data
    #     :param idx_test: index of testing data
    #     :param data: (segments, 1, channel, data)
    #     :param label: (segments,)
    #     :return: data and label
    #     """
    #     data_train = data[idx_train]
    #     label_train = label[idx_train]
    #     data_test = data[idx_test]
    #     label_test = label[idx_test]
    #     if self.args.dataset == 'Att' or self.args.dataset == 'DEAP':
    #         """
    #         For DEAP we want to do trial-wise 10-fold, so the idx_train/idx_test is for
    #         trials.
    #         data: (trial, segment, 1, chan, datapoint)
    #         To use the normalization function, we should change the dimension from
    #         (trial, segment, 1, chan, datapoint) to (trial*segments, 1, chan, datapoint)
    #         """
    #         data_train = np.concatenate(data_train, axis=0)
    #         label_train = np.concatenate(label_train, axis=0)
    #         if len(data_test.shape) > 4:
    #             """
    #             When leave one trial out is conducted, the test data will be (segments, 1, chan, datapoint), hence,
    #             no need to concatenate the first dimension to get trial*segments
    #             """
    #             data_test = np.concatenate(data_test, axis=0)
    #             label_test = np.concatenate(label_test, axis=0)
    #     data_train, data_test = self.normalize(train=data_train, test=data_test)
    #     # Prepare the data format for training the model using PyTorch
    #     data_train = torch.from_numpy(data_train).float()
    #     label_train = torch.from_numpy(label_train).long()
    #     data_test = torch.from_numpy(data_test).float()
    #     label_test = torch.from_numpy(label_test).long()
    #     return data_train, label_train, data_test, label_test


    def load_eeg_data(self):
        if self.data_using != 'DEAP':
            raise ValueError('This release trains on DEAP only.')
        dir = self.data_dir
        label_type = self.label_type
        dataset_ = np.load(dir + label_type + "_Type_DEAP_AllSub_Combined.npz")
        data_ = dataset_['data']
        lbls_ = dataset_['label']
        data_ = data_.reshape(32, 600, 32, 512)
        lbls_ = lbls_.reshape(32, 600)
        print(label_type + " labeltype shape:", data_.shape)
        print(label_type + " labeltype shape:", lbls_.shape)
        return data_, lbls_

    def normalize(self, train, test):
        """
        this function do standard normalization for EEG channel by channel
        :param train: training data (sample, 1, chan, datapoint)
        :param test: testing data (sample, 1, chan, datapoint)
        :return: normalized training and testing data
        """
        # data: sample x 1 x channel x data

        for channel in range(train.shape[1]):
            mean = np.mean(train[:, channel, :])
            std = np.std(train[:, channel, :])
            train[:, channel, :] = (train[:, channel, :] - mean) / std
            test[:, channel, :] = (test[:, channel, :] - mean) / std
        return train, test

    def split_balance_class(self, data, label, train_rate, random):
        """
        Get the validation set using the same percentage of the two classe samples
        :param data: training data (segment, 1, channel, data)
        :param label: (segments,)
        :param train_rate: the percentage of training data
        :param random: bool, whether to shuffle the training data before get the validation data
        :return: data_trian, label_train, and data_val, label_val
        """
        # Data dimension: segment x 1 x channel x data
        # Label dimension: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_random_0 = copy.deepcopy(index_0)

        # for class 1
        index_random_1 = copy.deepcopy(index_1)

        if random == True:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)

        index_train = np.concatenate((index_random_0[:int(len(index_random_0) * train_rate)],
                                      index_random_1[:int(len(index_random_1) * train_rate)]),
                                     axis=0)
        index_val = np.concatenate((index_random_0[int(len(index_random_0) * train_rate):],
                                    index_random_1[int(len(index_random_1) * train_rate):]),
                                   axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label


    def n_fold_CV(self, subject, fold=10):
        # Reset GE routing accumulator for this process-level run
        if str(getattr(self.args, 'ge_diag_routing', 'off')).lower() == 'on':
            try:
                from ge_bandgraph import GraphExpertMoE
                GraphExpertMoE.reset_global_routing_accum()
            except Exception:
                pass
        """
        this function achieves n-fold cross-validation
        :param subject: how many subject to load
        :param fold: how many fold
        """
        # Train and evaluate the model subject by subject
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1
        #subject: 0->32
        data, label = self.load_eeg_data()  # data:(32, 600, 32, 512)
        # P0: optional outer-fold subset (KFold split/seed unchanged; only skip folds)
        outer_folds = None
        if getattr(self.args, 'outer_folds', None) is not None and str(self.args.outer_folds).strip() != '':
            outer_folds = set(int(x.strip()) for x in str(self.args.outer_folds).split(',') if x.strip() != '')
            print(f'[P0] running only outer folds: {sorted(outer_folds)}')

        subject_ids_done = []
        n_test_folds = 0
        n_collapse_pos = 0
        n_collapse_neg = 0
        ttauc = []  # per-subject mean test AUC (fold-pooled via scores when available)
        subject_scores = []  # list of per-subject y_score arrays (for bootstrap CI)
        subject_labels = []  # list of per-subject y_true arrays
        subject_sample_idxs = []  # list of per-subject original trial indices
        for sub in subject:

            # data, label = self.load_per_subject(sub)
            # data, label = self.load_eeg_data(args) # data:(32, 600, 32, 512)
            data_ = data[int(sub)]
            label_ = label[int(sub)]
            va_val = Averager()
            vf_val = Averager()
            preds, acts = [], []
            scores = []
            sample_idxs = []  # original segment indices (KFold), aligned with acts/scores
            fold_aucs = []
            split_level = getattr(self.args, 'split_level', 'segment') or 'segment'
            print(f'[split] level={split_level} outer_folds={fold} seed={KFOLD_RANDOM_STATE}')
            for idx_fold, (idx_train, idx_test) in enumerate(
                    iter_outer_splits(len(data_), fold, split_level)):
                if outer_folds is not None and idx_fold not in outer_folds:
                    continue
                print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))
                idx_train = np.asarray(idx_train, dtype=np.int64)
                idx_test = np.asarray(idx_test, dtype=np.int64)
                if split_level == 'trial':
                    tr_trials = set(idx_train // SEG_PER_TRIAL_DEAP)
                    te_trials = set(idx_test // SEG_PER_TRIAL_DEAP)
                    assert tr_trials.isdisjoint(te_trials), (
                        f'trial leakage fold={idx_fold}: train∩test={tr_trials & te_trials}'
                    )
                    assert len(idx_test) == (N_TRIALS_DEAP // fold) * SEG_PER_TRIAL_DEAP, (
                        f'expected {(N_TRIALS_DEAP // fold) * SEG_PER_TRIAL_DEAP} test segs, got {len(idx_test)}'
                    )
                    assert len(te_trials) == N_TRIALS_DEAP // fold, (
                        f'expected {N_TRIALS_DEAP // fold} test trials, got {len(te_trials)}'
                    )
                    print(f'  [split-check] test_segs={len(idx_test)} test_trials={sorted(te_trials)} '
                          f'train∩test_trials=∅')
                data_train = data_[idx_train]
                label_train = label_[idx_train]
                data_test = data_[idx_test]
                label_test = label_[idx_test]
                # test DataLoader uses shuffle=False → score order == idx_test order
                sample_idxs.extend(list(np.asarray(idx_test, dtype=np.int64)))
                if self.data_norm == True:
                    data_train, data_test = self.normalize(train=data_train, test=data_test)
                # Change data to Tensor Format
                data_train = torch.from_numpy(data_train).float()
                label_train = torch.from_numpy(label_train).long()
                data_test = torch.from_numpy(data_test).float()
                label_test = torch.from_numpy(label_test).long()

                if self.args.reproduce:
                    # to reproduce the reported ACC
                    acc_test, pred, act, fold_info = test(args=self.args, data=data_test, label=label_test,
                                           reproduce=self.args.reproduce,
                                           subject=sub, fold=idx_fold)
                    acc_val = 0
                    f1_val = 0
                else:
                    # to train new models
                    acc_val, f1_val = self.first_stage(
                        data=data_train, label=label_train,
                        subject=sub, fold=idx_fold,
                        train_seg_indices=idx_train,
                    )
                    # LE probe: dump outer-train A/B/C/D from selected max-f1 (side-channel)
                    if str(getattr(self.args, 'le_probe_dump', 'off')).lower() == 'on':
                        try:
                            train_loader_probe = get_dataloader(
                                args=self.args, data=data_train, label=label_train,
                                batch_size=self.args.batch_size, shuffle=False,
                            )
                            model_p = get_model(self.args)
                            if CUDA:
                                model_p = model_p.cuda()
                            ckpt = self.args.load_path_final
                            if not osp.exists(ckpt):
                                ckpt = self.args.load_path
                            model_p.load_state_dict(torch.load(ckpt))
                            loss_fn_p = nn.CrossEntropyLoss()
                            _ = predict(
                                self.args, data_loader=train_loader_probe, net=model_p,
                                loss_fn=loss_fn_p, subject=sub, fold=idx_fold,
                                log_tag='probe_train',
                            )
                            del model_p
                        except Exception as e:
                            print(f'[le-probe-dump] outer-train dump failed: {e}')
                    # combine_train(args=self.args,
                    #               data=data_train, label=label_train,
                    #               subject=sub, fold=idx_fold, target_acc=1)
                    if self.cts:
                        combine_train(args=self.args,
                                      data=data_train, label=label_train,
                                      subject=sub, fold=idx_fold, target_acc=1)

                    acc_test, pred, act, fold_info = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=self.args.reproduce,
                                               subject=sub, fold=idx_fold)
                n_test_folds += 1
                if fold_info.get('collapse_pos'):
                    n_collapse_pos += 1
                if fold_info.get('collapse_neg'):
                    n_collapse_neg += 1
                fa = fold_info.get('auc', float('nan'))
                if fa == fa:  # not NaN
                    fold_aucs.append(float(fa))
                va_val.add(acc_val)
                vf_val.add(f1_val)
                preds.extend(pred)
                acts.extend(act)
                if fold_info.get('y_score') is not None:
                    scores.extend(fold_info['y_score'])

            tva.append(va_val.item())
            tvf.append(vf_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
            macro_f1, bal_acc = get_diag_metrics(y_pred=preds, y_true=acts)
            # Point AUC = AUC on fold-pooled subject scores (same estimator as bootstrap CI)
            fold_mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float('nan')
            if scores and acts:
                sub_auc = float(safe_roc_auc(acts, scores))
            else:
                sub_auc = fold_mean_auc
            auc_s = f'{sub_auc:.4f}' if sub_auc == sub_auc else 'nan'
            fold_note = (
                f' (fold_mean_AUC={fold_mean_auc:.4f})'
                if fold_mean_auc == fold_mean_auc else ''
            )
            print(f'[diag] sub={int(sub)} test_ACC={acc:.4f} test_F1={f1:.4f} '
                  f'macro_F1={macro_f1:.4f} balanced_ACC={bal_acc:.4f} '
                  f'AUC={auc_s}{fold_note}')
            tta.append(acc)
            ttf.append(f1)
            ttauc.append(sub_auc)
            subject_ids_done.append(int(sub))
            subject_scores.append(np.asarray(scores, dtype=np.float64))
            subject_labels.append(np.asarray(acts, dtype=np.int64))
            subject_sample_idxs.append(np.asarray(sample_idxs, dtype=np.int64))
            result = '{},{}'.format(tta[-1], f1)
            self.log2txt(result)
            self.log2txt(
                f'[diag] sub={int(sub)} macro_F1={macro_f1:.4f} balanced_ACC={bal_acc:.4f} AUC={auc_s}'
            )

        # prepare final report
        tta = np.array(tta) #total test acc
        ttf = np.array(ttf) #total test F1
        tva = np.array(tva) #total validation acc
        tvf = np.array(tvf) #total validation F1
        mACC = np.mean(tta)
        std = np.std(tta)
        mF1 = np.mean(ttf)
        F1_std = np.std(ttf)

        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)
        auc_arr = np.array([x for x in ttauc if x == x], dtype=np.float64)
        mAUC = float(np.mean(auc_arr)) if len(auc_arr) else float('nan')
        AUC_std = float(np.std(auc_arr)) if len(auc_arr) else float('nan')
        boot = bootstrap_mean_auc_ci(subject_labels, subject_scores, n_boot=1000, seed=8989)
        # Prefer bootstrap's point mean (identical estimator to CI); keep ttauc as check
        auc_ci_lo, auc_ci_hi = boot['ci_lo'], boot['ci_hi']
        auc_boot_se = boot['se']
        if boot['mean_auc'] == boot['mean_auc']:
            mAUC = float(boot['mean_auc'])
            AUC_std = float(np.std([a for a in boot['per_subject_aucs'] if a == a])) if boot['per_subject_aucs'] else float('nan')

        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: test mean F1:{} std:{}'.format(mF1, F1_std))
        print('Final: val mean F1:{}'.format(mF1_val))
        auc_msg = f'{mAUC:.4f} std={AUC_std:.4f}' if mAUC == mAUC else 'nan'
        print(f'Final: test mean AUC:{auc_msg}')
        if auc_ci_lo == auc_ci_lo:
            print(f'Final: test mean AUC bootstrap 95% CI=[{auc_ci_lo:.4f}, {auc_ci_hi:.4f}] '
                  f'SE={auc_boot_se:.4f} (n_boot=1000, per-subject resample then mean)')
            self.log2txt(
                f'[auc-bootstrap] mean={mAUC:.4f} 95%CI=[{auc_ci_lo:.4f},{auc_ci_hi:.4f}] '
                f'SE={auc_boot_se:.4f} method=per_subject_resample'
            )
        n_collapse = n_collapse_pos + n_collapse_neg
        print(f'[collapse-summary] test_folds={n_test_folds} '
              f'COLLAPSE-POS={n_collapse_pos} COLLAPSE-NEG={n_collapse_neg} '
              f'total_collapse={n_collapse} rate={n_collapse/max(n_test_folds,1):.3f}')
        self.log2txt(
            f'[collapse-summary] folds={n_test_folds} pos={n_collapse_pos} '
            f'neg={n_collapse_neg} total={n_collapse}'
        )
        results = 'test mAcc={} mAcc std={} mF1={} mF1 std={} mAUC={} mAUC std={} val mAcc={} val mF1={}'.format(
            mACC, std, mF1, F1_std, mAUC, AUC_std, mACC_val, mF1_val
        )
        self.log2txt(results)


        try:
            pred_dir = osp.join(self.args.save_path, 'pred_scores')
            ensure_path(pred_dir)
            tag = getattr(self.args, 'run_tag', '') or 'untagged'
            pred_path = osp.join(pred_dir, f'{tag}.npz')
            np.savez_compressed(
                pred_path,
                subject_ids=np.asarray(subject_ids_done, dtype=np.int64),
                test_accs=np.asarray(tta, dtype=np.float64),
                test_f1s=np.asarray(ttf, dtype=np.float64),
                test_aucs=np.asarray(ttauc, dtype=np.float64),
            )
            print(f'[pred-dump] saved aligned scores -> {pred_path}')
        except Exception as e:
            print(f'[pred-dump] failed: {e}')

    def first_stage(self, data, label, subject, fold, train_seg_indices=None):
        """
        Nested inner 3-fold CV on the outer-train split:
          1. fit on inner-train, score inner-val
          2. keep the checkpoint with the best inner-val metric
        Returns mean inner-val ACC and F1.
        """
        split_level = getattr(self.args, 'split_level', 'segment') or 'segment'
        va = Averager()
        vf = Averager()
        va_item = []
        vf_item = []
        maxAcc = -1.0
        maxF1 = -1.0
        maxAUC = -1.0
        select_mode = ckpt_select_mode(self.load_path)
        if split_level == 'trial':
            if train_seg_indices is None:
                raise ValueError('first_stage trial split requires train_seg_indices')
            seg_idx = np.asarray(train_seg_indices, dtype=np.int64)
            assert len(seg_idx) == len(data), 'train_seg_indices length mismatch'
            trial_ids = seg_idx // SEG_PER_TRIAL_DEAP
            unique_trials = np.unique(trial_ids)
            kf = KFold(n_splits=3, shuffle=True, random_state=KFOLD_RANDOM_STATE)
            inner_splits = []
            for tr_u, va_u in kf.split(unique_trials):
                tr_set = set(unique_trials[tr_u].tolist())
                va_set = set(unique_trials[va_u].tolist())
                assert tr_set.isdisjoint(va_set), 'inner trial leakage'
                idx_train = np.where(np.isin(trial_ids, list(tr_set)))[0]
                idx_val = np.where(np.isin(trial_ids, list(va_set)))[0]
                inner_splits.append((idx_train, idx_val))
        else:
            kf = KFold(n_splits=3, shuffle=True, random_state=KFOLD_RANDOM_STATE)
            inner_splits = list(kf.split(data))

        for i, (idx_train, idx_val) in enumerate(inner_splits):
            print('Inner 3-fold-CV Fold:{}'.format(i))
            data_train, label_train = data[idx_train], label[idx_train]
            data_val, label_val = data[idx_val], label[idx_val]
            fold_train_start = time.time()
            try:
                acc_val, F1_val, F1_max, AUC_max = train(args=self.args,
                                        data_train=data_train,
                                        label_train=label_train,
                                        data_val=data_val,
                                        label_val=label_val,
                                        subject=subject,
                                        fold=fold)
            except Exception as e:
                try:
                    from moe import DiagEapatchDone
                except Exception:
                    DiagEapatchDone = tuple()
                if DiagEapatchDone and isinstance(e, DiagEapatchDone):
                    print(f'[diag-eapatch] first_stage stop: {e}')
                    raise
                raise
            va.add(acc_val)
            vf.add(F1_val)
            va_item.append(acc_val)
            vf_item.append(F1_max)

            def _promote_candidate(metric_name, metric_value, ckpt_stem):
                old_name = osp.join(self.args.save_path, 'candidate.pth')
                new_name = osp.join(self.args.save_path, f'{ckpt_stem}.pth')
                if not os.path.exists(old_name):
                    raise RuntimeError(
                        f'[ckpt] candidate.pth missing before rename: '
                        f'sub={subject} outer_fold={fold} inner={i} {metric_name}={metric_value}'
                    )
                mtime = os.path.getmtime(old_name)
                if mtime < fold_train_start - 1.0:
                    raise RuntimeError(
                        f'[ckpt] candidate.pth is STALE '
                        f'(mtime={mtime:.1f} < train_start={fold_train_start:.1f}): '
                        f'sub={subject} outer_fold={fold} inner={i}'
                    )
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
                old_val = osp.join(self.args.save_path, 'candidate_val.npz')
                new_val = osp.join(self.args.save_path, f'{ckpt_stem}_val.npz')
                if os.path.exists(old_val):
                    if os.path.exists(new_val):
                        os.remove(new_val)
                    os.rename(old_val, new_val)
                print(f'New max {metric_name} model saved, with the val {metric_name} being:{metric_value}')

            if select_mode == 'f1' and F1_max > maxF1:
                maxF1 = F1_max
                _promote_candidate('F1', maxF1, 'max-f1')
            elif select_mode == 'acc' and acc_val > maxAcc:
                maxAcc = acc_val
                _promote_candidate('ACC', acc_val, 'max-acc')
            elif select_mode == 'auc':
                auc_ok = AUC_max == AUC_max
                if auc_ok and AUC_max > maxAUC:
                    maxAUC = AUC_max
                    _promote_candidate('AUC', maxAUC, 'max-auc')

        mAcc = va.item()
        mF1 = vf.item()
        return mAcc, mF1

    def log2txt(self, content):
        """
        this function log the content to results.txt
        :param content: string, the content to log
        """
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()
