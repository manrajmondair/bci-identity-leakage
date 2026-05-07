# Audit report — 2026-05-07T22:57:34Z

git: `58f8de423ecfa0f6164f1c0e14586b1e7903b2f0`
platform: `macOS-26.4.1-arm64-arm-64bit`
python: `3.11.14`

## Summary: 76 OK, 0 WARN, 0 FAIL

| | Check | Detail |
|---|---|---|
| ✓ | valid_subjects() count | actual=104 expected=104 |
| ✓ | known-bad subjects dropped | set() should be empty |
| ✓ | imagery & execution & baseline runs are disjoint families | imagery={4, 6, 8, 10, 12, 14}  execution={3, 5, 7, 9, 11, 13}  baseline={1, 2} |
| ✓ | X.shape[0] == y.shape[0] == subject_ids.shape[0] | X=(28080, 64, 320) y=(28080,) sids=(28080,) (actual=True expected=True) |
| ✓ | n_channels (PhysioNet) | actual=64 expected=64 |
| ✓ | n_times (2s @ 160Hz) | actual=320 expected=320 |
| ✓ | classes used | actual=[0, 1, 2, 3] expected=[0, 1, 2, 3] |
| ✓ | subject count in cached data | actual=104 expected=104 |
| ✓ | no NaN/Inf in X | X has 0 non-finite values |
| ✓ | per-subject window counts are tightly clustered | mean=270.0  std=0.0  min=270  max=270 |
| ✓ | each subject has all 4 motor-imagery classes, no severe imbalance | 0 suspect subjects |
| ✓ | windows-per-trial ≈ 3 (4-s trial, 2-s window, 1-s stride) | 9360 unique trials, 28080 windows, avg 3.000 per trial |
| ✓ | each trial_id maps to a single subject (no cross-subject collision) | 0 conflicts |
| ✓ | subject's trial_ids land inside its 100k-offset block | 0 mismatched |
| ✓ | channel order length | actual=64 expected=64 |
| ✓ | A1 train and test runs are disjoint | train={8, 10, 4, 6}  test={12, 14} |
| ✓ | A1 train and test runs are both subsets of imagery runs | missing from imagery: set() |
| ✓ | A1 closed-set: train_subjects == test_subjects | |train|=104 |test|=104 diff=set() |
| ✓ | A1 train and test trials are disjoint (no within-trial leakage) | |overlap|=0 |
| ✓ | A1 train run-set in train slice | actual={8, 10, 4, 6} expected={8, 10, 4, 6} |
| ✓ | A1 test run-set in test slice | actual={12, 14} expected={12, 14} |
| ✓ | A2: probe train (execution) ∩ probe test (imagery) is empty | train_runs=[3, 5, 7, 9, 11, 13]  test_runs=[12, 14] |
| ✓ | A3 sessions used: 0train -> 1test (different recording days) | by construction in experiments/05_a3_cross_session.py |
| ✓ | A4: train_subjects ∩ test_subjects is empty (open-set) | |train|=80 |test|=24 |overlap|=0 |
| ✓ | A4 |train| | actual=80 expected=80 |
| ✓ | A4 |test (held-out)| | actual=24 expected=24 |
| ✓ | A4 union == 104 valid subjects | missing=set()  extra=set() |
| ✓ | A1 attack: bootstrap CI uses grouped_bootstrap_ci over trial_ids | see attacks/closed_set.py |
| ✓ | A1 attack: probe trained ONLY on Z_train, tested ONLY on Z_test | fit/predict separation enforced |
| ✓ | A4 attack: training on (X_train, subj_train) only | see attacks/verification.py |
| ✓ | A4 attack: test pairs sampled from test_subjects only | verification pair sampling restricted to held-out subjects |
| ✓ | EEGNet uses input_scale=1e6 (volts -> microvolts) | see models/eegnet.py |
| ✓ | shuffled-label probe top-1 ≈ chance (0.0096) | top1=0.0100  top5=0.0472  chance=0.0096 |
| ✓ | same probe + REAL labels recovers identity well above chance | top1(real)=0.9482  vs chance=0.0096 |
| ✓ | results/02_closed_set_reid.json: non-empty list | 6 rows |
| ✓ | results/02_closed_set_reid.json: top1 is finite | eegnet/knn: top1=0.18173076923076922 |
| ✓ | results/02_closed_set_reid.json: top1_ci_low is finite | eegnet/knn: top1_ci_low=0.17158119658119658 |
| ✓ | results/02_closed_set_reid.json: top1_ci_high is finite | eegnet/knn: top1_ci_high=0.19316506410256412 |
| ✓ | results/02_closed_set_reid.json: chance_top1 is finite | eegnet/knn: chance_top1=0.009615384615384616 |
| ✓ | results/02_closed_set_reid.json: CI brackets the point estimate | eegnet/knn: low=0.1716 pt=0.1817 hi=0.1932 |
| ✓ | results/02_closed_set_reid.json: top1 is finite | eegnet/logreg: top1=0.41068376068376067 |
| ✓ | results/02_closed_set_reid.json: top1_ci_low is finite | eegnet/logreg: top1_ci_low=0.3971153846153846 |
| ✓ | results/02_closed_set_reid.json: top1_ci_high is finite | eegnet/logreg: top1_ci_high=0.4231864316239316 |
| ✓ | results/02_closed_set_reid.json: chance_top1 is finite | eegnet/logreg: chance_top1=0.009615384615384616 |
| ✓ | results/02_closed_set_reid.json: CI brackets the point estimate | eegnet/logreg: low=0.3971 pt=0.4107 hi=0.4232 |
| ✓ | results/02_closed_set_reid.json: top1 is finite | fbcsp_lda/knn: top1=0.8333333333333334 |
| ✓ | results/02_closed_set_reid.json: top1_ci_low is finite | fbcsp_lda/knn: top1_ci_low=0.8228605769230769 |
| ✓ | results/02_closed_set_reid.json: top1_ci_high is finite | fbcsp_lda/knn: top1_ci_high=0.8439129273504273 |
| ✓ | results/02_closed_set_reid.json: chance_top1 is finite | fbcsp_lda/knn: chance_top1=0.009615384615384616 |
| ✓ | results/02_closed_set_reid.json: CI brackets the point estimate | fbcsp_lda/knn: low=0.8229 pt=0.8333 hi=0.8439 |
| ✓ | results/02_closed_set_reid.json: top1 is finite | fbcsp_lda/logreg: top1=0.8905982905982905 |
| ✓ | results/02_closed_set_reid.json: top1_ci_low is finite | fbcsp_lda/logreg: top1_ci_low=0.8811965811965812 |
| ✓ | results/02_closed_set_reid.json: top1_ci_high is finite | fbcsp_lda/logreg: top1_ci_high=0.8995726495726496 |
| ✓ | results/02_closed_set_reid.json: chance_top1 is finite | fbcsp_lda/logreg: chance_top1=0.009615384615384616 |
| ✓ | results/02_closed_set_reid.json: CI brackets the point estimate | fbcsp_lda/logreg: low=0.8812 pt=0.8906 hi=0.8996 |
| ✓ | results/02_closed_set_reid.json: top1 is finite | riemann_ts_lr/knn: top1=0.9857905982905983 |
| ✓ | results/02_closed_set_reid.json: top1_ci_low is finite | riemann_ts_lr/knn: top1_ci_low=0.982051282051282 |
| ✓ | results/02_closed_set_reid.json: top1_ci_high is finite | riemann_ts_lr/knn: top1_ci_high=0.9888915598290599 |
| ✓ | results/02_closed_set_reid.json: chance_top1 is finite | riemann_ts_lr/knn: chance_top1=0.009615384615384616 |
| ✓ | results/02_closed_set_reid.json: CI brackets the point estimate | riemann_ts_lr/knn: low=0.9821 pt=0.9858 hi=0.9889 |
| ✓ | results/02_closed_set_reid.json: top1 is finite | riemann_ts_lr/logreg: top1=1.0 |
| ✓ | results/02_closed_set_reid.json: top1_ci_low is finite | riemann_ts_lr/logreg: top1_ci_low=1.0 |
| ✓ | results/02_closed_set_reid.json: top1_ci_high is finite | riemann_ts_lr/logreg: top1_ci_high=1.0 |
| ✓ | results/02_closed_set_reid.json: chance_top1 is finite | riemann_ts_lr/logreg: chance_top1=0.009615384615384616 |
| ✓ | results/02_closed_set_reid.json: CI brackets the point estimate | riemann_ts_lr/logreg: low=1.0000 pt=1.0000 hi=1.0000 |
| ✓ | results/06_a4_open_set.json: auc in [0,1] | auc=0.925368772 |
| ✓ | results/06_a4_open_set.json: auc_ci_low in [0,1] | auc_ci_low=0.9230891471299654 |
| ✓ | results/06_a4_open_set.json: auc_ci_high in [0,1] | auc_ci_high=0.9275958176544729 |
| ✓ | results/06_a4_open_set.json: eer in [0,1] | eer=0.13323999999999997 |
| ✓ | results/06_a4_open_set.json: AUC CI brackets point estimate | low=0.9231 pt=0.9254 hi=0.9276 |
| ✓ | A1 eegnet top-1 above chance (compresses identity but leaks) | top1=0.411  (chance=0.0096) |
| ✓ | A1 eegnet task acc above chance (>0.25 for 4-class) | task_acc=0.388  (verifies input_scale fix is working) |
| ✓ | A1 fbcsp_lda top-1 substantially above chance | top1=0.891 |
| ✓ | A1 riemann_ts_lr top-1 high (lit. closed-set EEG re-ID >= 0.95) | top1=1.000 |
| ✓ | A4 AUC plausible for open-set EEG verification (0.80 < AUC < 0.99) | AUC=0.925 |
| ✓ | A4 EER consistent with AUC (rough rule: EER ≈ 1 - AUC) | EER=0.133  1-AUC=0.075  |diff|=0.059 |
