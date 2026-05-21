# Audit report — 2026-05-21T03:33:37Z

git: `97cc07d2d93f0581f8335676464f8e23149d7d86`
platform: `macOS-26.4.1-arm64-arm-64bit`
python: `3.11.14`

## Summary: 273 OK, 0 WARN, 0 FAIL

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
| ✓ | 20_a3_lee2019: rows present | 6 rows |
| ✓ | 20_a3_lee2019 eegnet/knn top1 in [0,1] | value=0.14376543209876544 |
| ✓ | 20_a3_lee2019 eegnet/knn CI brackets top1 | low=0.13879552469135803 point=0.14376543209876544 high=0.14864274691358026 |
| ✓ | 20_a3_lee2019 eegnet/knn chance matches 1/54 | chance=0.01852 |
| ✓ | 20_a3_lee2019 eegnet/knn dataset tag set | dataset=lee2019 |
| ✓ | 20_a3_lee2019 eegnet/knn top1 above chance | top1=0.144  chance=0.019 |
| ✓ | 20_a3_lee2019 eegnet/logreg top1 in [0,1] | value=0.2666975308641975 |
| ✓ | 20_a3_lee2019 eegnet/logreg CI brackets top1 | low=0.26080246913580246 point=0.2666975308641975 high=0.27262345679012345 |
| ✓ | 20_a3_lee2019 eegnet/logreg chance matches 1/54 | chance=0.01852 |
| ✓ | 20_a3_lee2019 eegnet/logreg dataset tag set | dataset=lee2019 |
| ✓ | 20_a3_lee2019 eegnet/logreg top1 above chance | top1=0.267  chance=0.019 |
| ✓ | 20_a3_lee2019 fbcsp_lda/knn top1 in [0,1] | value=0.2556481481481481 |
| ✓ | 20_a3_lee2019 fbcsp_lda/knn CI brackets top1 | low=0.24929012345679014 point=0.2556481481481481 high=0.2625007716049383 |
| ✓ | 20_a3_lee2019 fbcsp_lda/knn chance matches 1/54 | chance=0.01852 |
| ✓ | 20_a3_lee2019 fbcsp_lda/knn dataset tag set | dataset=lee2019 |
| ✓ | 20_a3_lee2019 fbcsp_lda/knn top1 above chance | top1=0.256  chance=0.019 |
| ✓ | 20_a3_lee2019 fbcsp_lda/logreg top1 in [0,1] | value=0.2878703703703704 |
| ✓ | 20_a3_lee2019 fbcsp_lda/logreg CI brackets top1 | low=0.28101697530864195 point=0.2878703703703704 high=0.294445987654321 |
| ✓ | 20_a3_lee2019 fbcsp_lda/logreg chance matches 1/54 | chance=0.01852 |
| ✓ | 20_a3_lee2019 fbcsp_lda/logreg dataset tag set | dataset=lee2019 |
| ✓ | 20_a3_lee2019 fbcsp_lda/logreg top1 above chance | top1=0.288  chance=0.019 |
| ✓ | 20_a3_lee2019 riemann_ts_lr/knn top1 in [0,1] | value=0.6809259259259259 |
| ✓ | 20_a3_lee2019 riemann_ts_lr/knn CI brackets top1 | low=0.672991512345679 point=0.6809259259259259 high=0.688829475308642 |
| ✓ | 20_a3_lee2019 riemann_ts_lr/knn chance matches 1/54 | chance=0.01852 |
| ✓ | 20_a3_lee2019 riemann_ts_lr/knn dataset tag set | dataset=lee2019 |
| ✓ | 20_a3_lee2019 riemann_ts_lr/knn top1 above chance | top1=0.681  chance=0.019 |
| ✓ | 20_a3_lee2019 riemann_ts_lr/logreg top1 in [0,1] | value=0.7489506172839506 |
| ✓ | 20_a3_lee2019 riemann_ts_lr/logreg CI brackets top1 | low=0.7413256172839506 point=0.7489506172839506 high=0.7566975308641976 |
| ✓ | 20_a3_lee2019 riemann_ts_lr/logreg chance matches 1/54 | chance=0.01852 |
| ✓ | 20_a3_lee2019 riemann_ts_lr/logreg dataset tag set | dataset=lee2019 |
| ✓ | 20_a3_lee2019 riemann_ts_lr/logreg top1 above chance | top1=0.749  chance=0.019 |
| ✓ | 24_a4_lee2019_within_session auc in [0,1] | value=0.920084228 |
| ✓ | 24_a4_lee2019_within_session auc_ci_low in [0,1] | value=0.9177048710264689 |
| ✓ | 24_a4_lee2019_within_session auc_ci_high in [0,1] | value=0.9223964296563323 |
| ✓ | 24_a4_lee2019_within_session eer in [0,1] | value=0.16396 |
| ✓ | 24_a4_lee2019_within_session AUC CI brackets point | low=0.9177048710264689 point=0.920084228 high=0.9223964296563323 |
| ✓ | 24_a4_lee2019_within_session train/test subjects disjoint | |train|=40 |test|=14 |overlap|=0 |
| ✓ | 24_a4_lee2019_within_session AUC above chance | AUC=0.920 |
| ✓ | 24_a4_lee2019_cross_session auc in [0,1] | value=0.8675353648 |
| ✓ | 24_a4_lee2019_cross_session auc_ci_low in [0,1] | value=0.8645027827403796 |
| ✓ | 24_a4_lee2019_cross_session auc_ci_high in [0,1] | value=0.8705720371545917 |
| ✓ | 24_a4_lee2019_cross_session eer in [0,1] | value=0.2156 |
| ✓ | 24_a4_lee2019_cross_session AUC CI brackets point | low=0.8645027827403796 point=0.8675353648 high=0.8705720371545917 |
| ✓ | 24_a4_lee2019_cross_session train/test subjects disjoint | |train|=40 |test|=14 |overlap|=0 |
| ✓ | 24_a4_lee2019_cross_session AUC above chance | AUC=0.868 |
| ✓ | 25_a5_lee2019 auc present | auc=0.7873799725651578 |
| ✓ | 25_a5_lee2019 auc_ci_low present | auc_ci_low=0.6583662860855843 |
| ✓ | 25_a5_lee2019 auc_ci_high present | auc_ci_high=0.8956196153075535 |
| ✓ | 25_a5_lee2019 advantage present | advantage=0.4444444444444445 |
| ✓ | 25_a5_lee2019 advantage_threshold present | advantage_threshold=0.424884410750819 |
| ✓ | 25_a5_lee2019 AUC CI brackets | low=0.6583662860855843 point=0.7873799725651578 high=0.8956196153075535 |
| ✓ | 25_a5_lee2019 members + non-members = total | M=27 + N=27 vs total=54 |
| ✓ | 26_a4_xds_iv2a_to_physionet auc in [0,1] | value=0.8308684895999999 |
| ✓ | 26_a4_xds_iv2a_to_physionet auc_ci_low in [0,1] | value=0.8274224505999254 |
| ✓ | 26_a4_xds_iv2a_to_physionet auc_ci_high in [0,1] | value=0.8343705316134441 |
| ✓ | 26_a4_xds_iv2a_to_physionet eer in [0,1] | value=0.24517999999999998 |
| ✓ | 26_a4_xds_iv2a_to_physionet AUC CI brackets | low=0.8274224505999254 point=0.8308684895999999 high=0.8343705316134441 |
| ✓ | 26_a4_xds_iv2a_to_physionet common channels non-empty | common channels = 22 |
| ✓ | 26_a4_xds_iv2a_to_physionet target sfreq = 160 Hz | sfreq=160.0 |
| ✓ | 26_a4_xds_physionet_to_lee2019 auc in [0,1] | value=0.8258127376 |
| ✓ | 26_a4_xds_physionet_to_lee2019 auc_ci_low in [0,1] | value=0.8222550200013051 |
| ✓ | 26_a4_xds_physionet_to_lee2019 auc_ci_high in [0,1] | value=0.829161223949675 |
| ✓ | 26_a4_xds_physionet_to_lee2019 eer in [0,1] | value=0.24685999999999997 |
| ✓ | 26_a4_xds_physionet_to_lee2019 AUC CI brackets | low=0.8222550200013051 point=0.8258127376 high=0.829161223949675 |
| ✓ | 26_a4_xds_physionet_to_lee2019 common channels non-empty | common channels = 48 |
| ✓ | 26_a4_xds_physionet_to_lee2019 target sfreq = 160 Hz | sfreq=160.0 |
| ✓ | 26_a4_xds_lee2019_to_physionet auc in [0,1] | value=0.4959382008 |
| ✓ | 26_a4_xds_lee2019_to_physionet auc_ci_low in [0,1] | value=0.4915516308707607 |
| ✓ | 26_a4_xds_lee2019_to_physionet auc_ci_high in [0,1] | value=0.5002237930107989 |
| ✓ | 26_a4_xds_lee2019_to_physionet eer in [0,1] | value=0.50524 |
| ✓ | 26_a4_xds_lee2019_to_physionet AUC CI brackets | low=0.4915516308707607 point=0.4959382008 high=0.5002237930107989 |
| ✓ | 26_a4_xds_lee2019_to_physionet common channels non-empty | common channels = 48 |
| ✓ | 26_a4_xds_lee2019_to_physionet target sfreq = 160 Hz | sfreq=160.0 |
| ✓ | 26_a4_xds_iv2a_to_lee2019 auc in [0,1] | value=0.6725237848 |
| ✓ | 26_a4_xds_iv2a_to_lee2019 auc_ci_low in [0,1] | value=0.6680149572704587 |
| ✓ | 26_a4_xds_iv2a_to_lee2019 auc_ci_high in [0,1] | value=0.6772248083978092 |
| ✓ | 26_a4_xds_iv2a_to_lee2019 eer in [0,1] | value=0.3787 |
| ✓ | 26_a4_xds_iv2a_to_lee2019 AUC CI brackets | low=0.6680149572704587 point=0.6725237848 high=0.6772248083978092 |
| ✓ | 26_a4_xds_iv2a_to_lee2019 common channels non-empty | common channels = 21 |
| ✓ | 26_a4_xds_iv2a_to_lee2019 target sfreq = 160 Hz | sfreq=160.0 |
| ✓ | 27_d3_dp_aware_mia AUC in [0,1] | value=0.8909023668639053 |
| ✓ | 27_d3_dp_aware_mia AUC CI brackets | low=0.8255407296351823 point=0.8909023668639053 high=0.9434003359062322 |
| ✓ | 27_d3_dp_aware_mia final eps close to target | target=3.0  final=2.9962671959798315 |
| ✓ | 27_d3_dp_aware_mia delta is 1e-5 | delta=1e-05 |
| ✓ | 27_d3_dp_aware_mia_eps1.0 AUC in [0,1] | value=0.5059171597633136 |
| ✓ | 27_d3_dp_aware_mia_eps1.0 AUC CI brackets | low=0.3932389937106918 point=0.5059171597633136 high=0.6144494575936883 |
| ✓ | 27_d3_dp_aware_mia_eps1.0 final eps close to target | target=1.0  final=0.998405297625343 |
| ✓ | 27_d3_dp_aware_mia_eps1.0 delta is 1e-5 | delta=1e-05 |
| ✓ | 27_d3_dp_aware_mia_eps0.5 AUC in [0,1] | value=0.44933431952662733 |
| ✓ | 27_d3_dp_aware_mia_eps0.5 AUC CI brackets | low=0.334440451805174 point=0.44933431952662733 high=0.5584135083868453 |
| ✓ | 27_d3_dp_aware_mia_eps0.5 final eps close to target | target=0.5  final=0.4979112907196195 |
| ✓ | 27_d3_dp_aware_mia_eps0.5 delta is 1e-5 | delta=1e-05 |
| ✓ | 28_d3_model_inversion [no_defense] rank1 in [0,1] | value=0.0 |
| ✓ | 28_d3_model_inversion [no_defense] rank5 in [0,1] | value=0.1 |
| ✓ | 28_d3_model_inversion [no_defense] n_reconstructions = n_targets | n_recon=10 n_targets=10 |
| ✓ | 28_d3_model_inversion [dp_eps=3.0] rank1 in [0,1] | value=0.0 |
| ✓ | 28_d3_model_inversion [dp_eps=3.0] rank5 in [0,1] | value=0.1 |
| ✓ | 28_d3_model_inversion [dp_eps=3.0] n_reconstructions = n_targets | n_recon=10 n_targets=10 |
| ✓ | 29_d3_eps_sweep pareto non-empty | 5 rows |
| ✓ | 29_d3_eps_sweep[eps_0.5] final eps close to target | target=0.5 final=0.49526641739915667 |
| ✓ | 29_d3_eps_sweep[eps_0.5] task acc in [0,1] | value=0.25064102564102564 |
| ✓ | 29_d3_eps_sweep[eps_0.5] logreg top1 in [0,1] | value=0.027243589743589744 |
| ✓ | 29_d3_eps_sweep[eps_0.5] fine-tune top1 in [0,1] | value=0.04326923076923077 |
| ✓ | 29_d3_eps_sweep[eps_0.5] logreg CI brackets | low=0.023183760683760685 point=0.027243589743589744 high=0.030878739316239315 |
| ✓ | 29_d3_eps_sweep[eps_0.5] fine-tune CI brackets | low=0.03867254273504274 point=0.04326923076923077 high=0.04786324786324787 |
| ✓ | 29_d3_eps_sweep[eps_1.0] final eps close to target | target=1.0 final=0.9979799056164619 |
| ✓ | 29_d3_eps_sweep[eps_1.0] task acc in [0,1] | value=0.25256410256410255 |
| ✓ | 29_d3_eps_sweep[eps_1.0] logreg top1 in [0,1] | value=0.025106837606837608 |
| ✓ | 29_d3_eps_sweep[eps_1.0] fine-tune top1 in [0,1] | value=0.07008547008547009 |
| ✓ | 29_d3_eps_sweep[eps_1.0] logreg CI brackets | low=0.02125801282051282 point=0.025106837606837608 high=0.029062499999999998 |
| ✓ | 29_d3_eps_sweep[eps_1.0] fine-tune CI brackets | low=0.06431356837606837 point=0.07008547008547009 high=0.07596420940170939 |
| ✓ | 29_d3_eps_sweep[eps_3.0] final eps close to target | target=3.0 final=2.996188241361091 |
| ✓ | 29_d3_eps_sweep[eps_3.0] task acc in [0,1] | value=0.2931623931623932 |
| ✓ | 29_d3_eps_sweep[eps_3.0] logreg top1 in [0,1] | value=0.029594017094017093 |
| ✓ | 29_d3_eps_sweep[eps_3.0] fine-tune top1 in [0,1] | value=0.1361111111111111 |
| ✓ | 29_d3_eps_sweep[eps_3.0] logreg CI brackets | low=0.025961538461538463 point=0.029594017094017093 high=0.03344017094017094 |
| ✓ | 29_d3_eps_sweep[eps_3.0] fine-tune CI brackets | low=0.12788194444444442 point=0.1361111111111111 high=0.14487179487179488 |
| ✓ | 29_d3_eps_sweep[eps_10.0] final eps close to target | target=10.0 final=9.99157606572411 |
| ✓ | 29_d3_eps_sweep[eps_10.0] task acc in [0,1] | value=0.30106837606837605 |
| ✓ | 29_d3_eps_sweep[eps_10.0] logreg top1 in [0,1] | value=0.03258547008547009 |
| ✓ | 29_d3_eps_sweep[eps_10.0] fine-tune top1 in [0,1] | value=0.18878205128205128 |
| ✓ | 29_d3_eps_sweep[eps_10.0] logreg CI brackets | low=0.028952991452991454 point=0.03258547008547009 high=0.03653846153846154 |
| ✓ | 29_d3_eps_sweep[eps_10.0] fine-tune CI brackets | low=0.17894764957264958 point=0.18878205128205128 high=0.1987179487179487 |
| ✓ | 29_d3_eps_sweep[no_dp] task acc in [0,1] | value=0.3079059829059829 |
| ✓ | 29_d3_eps_sweep[no_dp] logreg top1 in [0,1] | value=0.032158119658119655 |
| ✓ | 29_d3_eps_sweep[no_dp] fine-tune top1 in [0,1] | value=0.15267094017094018 |
| ✓ | 29_d3_eps_sweep[no_dp] logreg CI brackets | low=0.028525641025641025 point=0.032158119658119655 high=0.0360042735042735 |
| ✓ | 29_d3_eps_sweep[no_dp] fine-tune CI brackets | low=0.14401442307692305 point=0.15267094017094018 high=0.16153846153846155 |
| ✓ | 30_theory_scaling cohort grid is increasing | cohort_grid=[10, 20, 40, 60, 80, 104] |
| ✓ | 30_theory_scaling eegnet N=10 top1 | value=0.8333333333333334 |
| ✓ | 30_theory_scaling eegnet N=20 top1 | value=0.7016666666666667 |
| ✓ | 30_theory_scaling eegnet N=40 top1 | value=0.5536111111111112 |
| ✓ | 30_theory_scaling eegnet N=60 top1 | value=0.5225925925925926 |
| ✓ | 30_theory_scaling eegnet N=80 top1 | value=0.44583333333333336 |
| ✓ | 30_theory_scaling eegnet N=104 top1 | value=0.41025641025641024 |
| ✓ | 30_theory_scaling riemann N=10 top1 | value=1.0 |
| ✓ | 30_theory_scaling riemann N=20 top1 | value=1.0 |
| ✓ | 30_theory_scaling riemann N=40 top1 | value=1.0 |
| ✓ | 30_theory_scaling riemann N=60 top1 | value=1.0 |
| ✓ | 30_theory_scaling riemann N=80 top1 | value=0.9995833333333334 |
| ✓ | 30_theory_scaling riemann N=104 top1 | value=1.0 |
| ✓ | 30_theory_scaling fine-tune <= Yeom bound (eps=0.5) | emp=0.043  bound=0.391 |
| ✓ | 30_theory_scaling fine-tune <= Yeom bound (eps=1.0) | emp=0.070  bound=0.631 |
| ✓ | 30_theory_scaling fine-tune <= Yeom bound (eps=3.0) | emp=0.136  bound=0.950 |
| ✓ | 30_theory_scaling fine-tune <= Yeom bound (eps=10.0) | emp=0.189  bound=1.000 |
| ✓ | 30_theory_scaling fine-tune <= Yeom bound (eps=None) | emp=0.153  bound=1.000 |
| ✓ | 31_federated_dp task acc in [0,1] | value=0.24946581196581197 |
| ✓ | 31_federated_dp logreg top1 in [0,1] | value=0.04401709401709402 |
| ✓ | 31_federated_dp fine-tune top1 in [0,1] | value=0.09615384615384616 |
| ✓ | 31_federated_dp rdp epsilon non-negative | eps_rdp=97.73299079819824 |
| ✓ | 31_federated_dp delta is 1e-5 | delta=1e-05 |
| ✓ | 32_fairness_lee2019[fbcsp] task acc in [0,1] | value=0.605925925925926 |
| ✓ | 32_fairness_lee2019[fbcsp] heterogeneity.mean in [0,1] | value=0.8026268015537205 |
| ✓ | 32_fairness_lee2019[fbcsp] heterogeneity.decile_gap in [0,1] | value=0.31728324532199026 |
| ✓ | 32_fairness_lee2019[fbcsp] heterogeneity.iqr in [0,1] | value=0.1701896194083694 |
| ✓ | 32_fairness_lee2019[fbcsp] heterogeneity.min in [0,1] | value=0.49765258215962443 |
| ✓ | 32_fairness_lee2019[fbcsp] heterogeneity.max in [0,1] | value=0.9656862745098039 |
| ✓ | 32_fairness_lee2019[fbcsp] min <= mean <= max | min=0.498  mean=0.803  max=0.966 |
| ✓ | 32_fairness_lee2019[riemann] task acc in [0,1] | value=0.6728703703703703 |
| ✓ | 32_fairness_lee2019[riemann] heterogeneity.mean in [0,1] | value=0.9994894584674238 |
| ✓ | 32_fairness_lee2019[riemann] heterogeneity.decile_gap in [0,1] | value=0.0 |
| ✓ | 32_fairness_lee2019[riemann] heterogeneity.iqr in [0,1] | value=0.0 |
| ✓ | 32_fairness_lee2019[riemann] heterogeneity.min in [0,1] | value=0.9873417721518988 |
| ✓ | 32_fairness_lee2019[riemann] heterogeneity.max in [0,1] | value=1.0 |
| ✓ | 32_fairness_lee2019[riemann] min <= mean <= max | min=0.987  mean=0.999  max=1.000 |
| ✓ | 32_fairness_lee2019[eegnet] task acc in [0,1] | value=0.7157407407407408 |
| ✓ | 32_fairness_lee2019[eegnet] heterogeneity.mean in [0,1] | value=0.4803884063986961 |
| ✓ | 32_fairness_lee2019[eegnet] heterogeneity.decile_gap in [0,1] | value=0.49001030749512275 |
| ✓ | 32_fairness_lee2019[eegnet] heterogeneity.iqr in [0,1] | value=0.2500987760481431 |
| ✓ | 32_fairness_lee2019[eegnet] heterogeneity.min in [0,1] | value=0.14767932489451477 |
| ✓ | 32_fairness_lee2019[eegnet] heterogeneity.max in [0,1] | value=0.9166666666666666 |
| ✓ | 32_fairness_lee2019[eegnet] min <= mean <= max | min=0.148  mean=0.480  max=0.917 |
| ✓ | 33_asymmetry_mechanism auc in [0,1] | value=0.501403552 |
| ✓ | 33_asymmetry_mechanism auc_ci_low in [0,1] | value=0.4985621280548043 |
| ✓ | 33_asymmetry_mechanism auc_ci_high in [0,1] | value=0.5042560792535095 |
| ✓ | 33_asymmetry_mechanism eer in [0,1] | value=0.49928 |
| ✓ | 33_asymmetry_mechanism AUC CI brackets | low=0.4985621280548043 point=0.501403552 high=0.5042560792535095 |
| ✓ | 33_asymmetry_mechanism synthetic label histogram has 4 classes | classes=['0', '1', '2', '3'] |
| ✓ | 33_asymmetry_mechanism hypothesis_supported is a bool | hypothesis_supported=False |
| ✓ | 34_tier1_multi_seed[a3_lee2019] eegnet_logreg_top1 has 3+ seeds | n=5 |
| ✓ | 34_tier1_multi_seed[a3_lee2019] eegnet_logreg_top1 std non-negative | std=0.01497261955134215 |
| ✓ | 34_tier1_multi_seed[a3_lee2019] fbcsp_logreg_top1 has 3+ seeds | n=5 |
| ✓ | 34_tier1_multi_seed[a3_lee2019] fbcsp_logreg_top1 std non-negative | std=0.0 |
| ✓ | 34_tier1_multi_seed[a3_lee2019] riemann_logreg_top1 has 3+ seeds | n=5 |
| ✓ | 34_tier1_multi_seed[a3_lee2019] riemann_logreg_top1 std non-negative | std=0.0 |
| ✓ | 34_tier1_multi_seed[a4_lee2019] auc_within_session has 3+ seeds | n=5 |
| ✓ | 34_tier1_multi_seed[a4_lee2019] auc_within_session std non-negative | std=0.03065317554486131 |
| ✓ | 34_tier1_multi_seed[a4_lee2019] eer_within_session has 3+ seeds | n=5 |
| ✓ | 34_tier1_multi_seed[a4_lee2019] eer_within_session std non-negative | std=0.03383151075550722 |
| ✓ | 34_tier1_multi_seed[xds_iv2a_to_physionet] auc has 3+ seeds | n=5 |
| ✓ | 34_tier1_multi_seed[xds_iv2a_to_physionet] auc std non-negative | std=0.015401887267999521 |
| ✓ | 34_tier1_multi_seed[xds_physionet_to_lee2019] auc has 3+ seeds | n=5 |
| ✓ | 34_tier1_multi_seed[xds_physionet_to_lee2019] auc std non-negative | std=0.036049352278796055 |
| ✓ | 34_tier1_multi_seed[xds_lee2019_to_physionet] auc has 3+ seeds | n=5 |
| ✓ | 34_tier1_multi_seed[xds_lee2019_to_physionet] auc std non-negative | std=0.005427965273293079 |
| ✓ | 34_tier1_multi_seed[xds_iv2a_to_lee2019] auc has 3+ seeds | n=5 |
| ✓ | 34_tier1_multi_seed[xds_iv2a_to_lee2019] auc std non-negative | std=0.015338908234493475 |
