# Demographic-axis fairness audit across the project's three datasets

The milestone reports sex- and age-stratified A1 attack accuracy on
PhysioNet (97 sex-known, 91 age-known subjects after the OpenNeuro
demographic recovery). Reviewers will reasonably ask which OTHER
demographic axes the benchmark could audit, and where the answer is
"the dataset doesn't publish it." This document is that audit.

## Headline finding

**No public motor-imagery EEG dataset that the project uses publishes
race or ethnicity per subject.** This is true across:

  - PhysioNet EEG-MMIDB (BCI2000 source, 109 subjects)
  - BCI Competition IV-2a (Graz, 9 subjects)
  - Lee 2019 OpenBMI (Korea University, 54 subjects)

The OpenNeuro `ds004362` BIDS conversion of the PhysioNet recordings
also does not publish race / ethnicity / ancestry; it adds Gender,
Age, and Handedness only.

This is itself a finding the paper should surface: the published
EEG-biometric literature operates in an axis-blind regime on what
medical-imaging-fairness work (e.g. Larrazabal 2020, Seyyed-Kalantari
2021) has shown to be a high-risk axis. EEG signal characteristics
correlate with skull conductivity and head geometry, which have been
linked to ancestry in the EEG-neurophysiology literature; if those
properties carry biometric-discriminative information, the privacy
threat could be unequally distributed by ancestry, but the data needed
to audit that hypothesis does not exist in current public corpora.

## Per-dataset audit table

| Dataset | Sex | Age | Handedness | Race/ethnicity | Education | Native lang. | Skull / head geometry |
|---|---|---|---|---|---|---|---|
| PhysioNet EEG-MMIDB | ✗ in source; ✓ via OpenNeuro ds004362 (95/109) | ✓ via OpenNeuro (95/109) | ✓ via OpenNeuro (95/109) | ✗ never | ✗ | ✗ | ✗ |
| BCI Competition IV-2a | cohort total only | age range only (22–30) | cohort total (8R/1L) | ✗ | ✗ | ✗ | ✗ |
| Lee 2019 OpenBMI | cohort total only (25 F / 29 M) | cohort range only (24–35) | not published | ✗ (recruited from a single university, all Korean cohort) | ✗ | ✗ | ✗ |

Legend: `✓ via X` means the metadata is recoverable from a sibling
release; `✗` means the dataset does not publish that axis at all.

## What we can audit, and where

Translated to experiments in this repo:

  - PhysioNet: sex and age stratification on A1 attack accuracy across
    all three victim families (`experiments/17_subgroup_fairness_eegnet.py`,
    `tools/subgroup_fairness.py`). Results shipped in milestone §3.6
    and extension §4.5 / §4.9.

  - IV-2a: no per-subject demographics published, so demographic
    stratification on A3 / cross-session is not possible. We
    document this explicitly in the limitations rather than report
    point estimates that are not power-supported.

  - Lee 2019: only cohort-level aggregates are public (54 subjects,
    25 F / 29 M, ages 24–35, 38 naive BCI users). Per-subject
    demographics are NOT in the paper or the GigaDB dataset release.
    `experiments/32_fairness_lee2019.py` therefore reports two things:
    (a) per-subject A1 heterogeneity (decile gap, percentile spread,
    Mann-Whitney U on the cross-subject distribution), which does not
    require per-subject demographics, and (b) demographic
    stratification *if* the user manually populates
    `data/external/lee2019_demographics.tsv` from a future per-subject
    release. The single-ethnicity nature of the cohort (all Korean
    university students from one institution) means race / ethnicity
    stratification within Lee 2019 is moot — the cohort is itself a
    single demographic stratum.

## Cross-dataset implication

The current biometric-leakage finding — that EEG-as-biometric works
at AUC = 0.925 on PhysioNet's 24 unseen subjects — has been validated
across the cross-dataset symmetric A4 experiments (PhysioNet ↔ IV-2a ↔
Lee 2019, experiment 26). The three cohorts together span:

  - PhysioNet: mixed-sex US adults, ages 19–67, BCI2000 protocol
  - IV-2a: small cohort, mostly young adults (22–30), Graz protocol
  - Lee 2019: Korean university students, ages 24–35, OpenBMI protocol

None of these probe biometric strength on, e.g., elderly
populations (>65), clinical populations (schizophrenia, epilepsy,
stroke), pediatric (<18), or non-East-Asian / non-European cohorts.
A reviewer concerned that EEG biometric strength varies by ancestry,
neurodevelopmental stage, or clinical status should regard our
results as conditional on the well-described demographic envelope
that these three datasets sample together.

## Recommendation for future dataset releases

The single highest-leverage action toward fair, regulator-usable
EEG-biometric benchmarks is publication of per-subject ancestry
metadata alongside upcoming BCI / motor-imagery corpora, with
appropriate consent — preferably as a separately licensed
demographic-axis side-file so signal-processing teams can audit
biometric / clinical fairness without rebuilding the participant
recruitment pipeline.

## References

- Larrazabal et al. (2020). *Gender imbalance in medical imaging
  datasets produces biased classifiers for computer-aided
  diagnosis.* PNAS.
- Seyyed-Kalantari et al. (2021). *Underdiagnosis bias of artificial
  intelligence algorithms applied to chest radiographs in under-
  served patient populations.* Nature Medicine.
- Lee et al. (2019). *EEG dataset and OpenBMI toolbox for three
  BCI paradigms: an investigation into BCI illiteracy.* GigaScience.
- Brunner et al. (2008). *BCI Competition 2008 — Graz data set A.*
  Graz University of Technology technical report.
- James (2023). *EEG Motor Movement/Imagery Dataset.* OpenNeuro
  ds004362.
