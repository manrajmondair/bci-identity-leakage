# `data/external/` — externally-sourced data referenced by analysis scripts

This directory holds small, third-party data files that the project's
analysis depends on but doesn't generate itself. Each file is committed
verbatim from its public source; provenance is documented here.

## `openneuro_ds004362_participants.tsv`

**Source.** OpenNeuro dataset [ds004362](https://openneuro.org/datasets/ds004362) — a BIDS conversion of the same BCI2000 source recordings PhysioNet hosts as the EEG Motor Movement/Imagery Database. Same `sub-001` ↔ `S001` indexing.

**Why we need it.** PhysioNet's EDF release scrubs subject metadata (`his_id: X, sex: 0` for every header), making demographic-stratified fairness analysis impossible on the canonical release. The OpenNeuro BIDS conversion republishes Gender + Age + Handedness for 95 of the 109 subjects with the same indexing. We map demographics back onto the PhysioNet release for the fairness analysis (`tools/subgroup_fairness.py`).

**Schema.** Tab-separated, 4 columns:

| Column | Values |
|---|---|
| `participant_id` | `sub-001` … `sub-109` |
| `Gender` | `M`, `F`, or `n/a` |
| `Age` | integer years, or `n/a` |
| `Handedness` | `R`, `L`, or `n/a` |

**Coverage in our 104-subject analysis cohort** (PhysioNet IDs 1–109 minus the project's drop list 88, 89, 92, 100, 104):

- 41 male / 56 female / 7 with unknown sex
- 91 with known age, range 19–67, median 38
- The remaining 13 subjects with unknown age are concentrated in the post-anonymization tail (sub-105 through sub-109 are entirely n/a) plus a small handful with partial coverage

**Known data quality issue.** Row `sub-044` has `Gender=n/a, Age='M', Handedness=n/a` — `'M'` is a transcription error in the OpenNeuro release (probably a column-shift in the original entry). Our loader (`tools.subgroup_fairness.load_demographics`) coerces non-integer ages to None, so `sub-044`'s row contributes to the unknown-age count rather than corrupting the age tertile cuts.

**Refresh.** To re-download from source:

```bash
curl -s https://s3.amazonaws.com/openneuro.org/ds004362/participants.tsv \
     -o data/external/openneuro_ds004362_participants.tsv
```

The committed file is small (~3 KB, 110 lines) so it's tracked in git rather than refetched on each clone — that way the analysis pipeline doesn't depend on OpenNeuro's S3 endpoint being reachable.

**License.** OpenNeuro data is released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/), so reproduction here is unrestricted.

## `physionet_duration_fingerprint.json`

**Purpose.** Per-subject per-run recording-duration vector for the 104
subjects in the analysis cohort, across all 14 PhysioNet runs. Used to
empirically check the OpenNeuro `sub-XXX` ↔ PhysioNet `S{XXX}`
indexing mapping when anyone downloads the OpenNeuro BIDS conversion.
Produced once locally; documented in [`MAPPING_VERIFICATION.md`](MAPPING_VERIFICATION.md).

## `MAPPING_VERIFICATION.md`

Documents the verification protocol and limits for the OpenNeuro →
PhysioNet indexing assumption. PhysioNet EDF headers anonymize
subject metadata so a strict per-subject empirical verification is
not possible from the local cache alone; the document describes the
duration-fingerprint check (point-wise verifiable for 7 of 104
subjects, cluster-membership verifiable for the rest) and the
structural argument that supports the assumed mapping.

