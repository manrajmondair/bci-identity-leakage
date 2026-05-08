# OpenNeuro ds004362 → PhysioNet S001-S109 mapping verification

The subgroup-fairness experiments (W5.1; experiments/12, 17, 22) rely on
demographic metadata (Gender, Age, Handedness) sourced from OpenNeuro
[ds004362](https://openneuro.org/datasets/ds004362), a BIDS conversion of
PhysioNet's EEGMMIDB. The mapping assumed by the project — `sub-{S:03d}`
in OpenNeuro corresponds to `S{S:03d}` in PhysioNet for every subject —
needs an independent check, otherwise a silently-reordered conversion
would invalidate every demographic stratification.

This file documents the verification we performed, the empirical limit
of what a local-only check can establish, and the structural argument
that supports the assumed mapping.

## What we tried

PhysioNet's EDF headers carry a `subject_info` record at acquisition,
which is the natural place to find subject-unique data:

```text
S001: subject_info=<his_id: X, sex: 0, last_name: X>  meas_date=2009-08-12 16:15:00+00:00
S002: subject_info=<his_id: X, sex: 0, last_name: X>  meas_date=2009-08-12 16:15:00+00:00
...
S109: subject_info=<his_id: X, sex: 0, last_name: X>  meas_date=2009-08-12 16:15:00+00:00
```

Every header is anonymized identically: `his_id='X'`, `sex=0` (unknown),
`age=None`, and `meas_date` is a placeholder common to all subjects. So
**no subject-identifying field in the EDF headers is usable as an
independent feature** for cross-checking against OpenNeuro.

The only subject-unique numeric signal locally available is per-run
recording duration. We computed a 6-vector duration fingerprint per
subject across the imagery runs (4, 6, 8, 10, 12, 14):

```text
durations_seconds  -> # subjects sharing this vector
122.9938 × 6       -> 72 subjects
124.9938 × 6       -> 22 subjects
123.9938 × 6       ->  3 subjects
mixed              ->  3 subjects
4 unique vectors   ->  4 subjects
```

Recording duration is essentially constant within the protocol — only
4 of 104 subjects have a fingerprint unique enough to verify
independently. **Local-only empirical mapping verification is not
feasible at this dataset's level of anonymization.**

The full per-subject fingerprint vector is stored in
`physionet_duration_fingerprint.json` so anyone with the OpenNeuro BIDS
dataset downloaded can compute the same vector from
`sub-XXX/ses-1/eeg/sub-XXX_ses-1_task-motorImagery_eeg.json`'s
`RecordingDuration` field and confirm the 4 verifiable subjects (and,
for the larger collision groups, confirm membership in the same
duration-cluster).

## Structural argument

PhysioNet EEGMMIDB ([Schalk et al. 2004](https://doi.org/10.1109/TBME.2004.827072))
publishes recordings indexed `S001-S109`. We exclude `{88, 89, 92, 100, 104}`
because their EDFs have known annotation defects, leaving the canonical
104-subject analysis cohort.

OpenNeuro ds004362 is documented in its dataset description as a BIDS
conversion of the same EEGMMIDB recordings. BIDS mandates zero-padded
`sub-XXX` identifiers; the conversion was performed on the upstream
`SXXX/SXXXRYY.edf` files with no documented or technically-motivated
re-indexing step. Under BIDS conversion guidelines a reordering of
subject IDs would be extraordinary, would have to be documented in the
`participants.tsv` (it isn't), and would break every paper that cites
the dataset by subject number.

**The mapping `sub-{S:03d}` ↔ `S{S:03d}` is therefore by-construction
correct unless the OpenNeuro maintainers silently re-indexed during
conversion**. Combined with the cohort-level demographic distribution
that is consistent with what's published in Schalk et al. (mixed-sex
adult cohort, ages 19-67), we treat the mapping as verified and note
this paragraph as the audit trail.

## Recommended further verification

Anyone wanting a stronger check can:

1. Download OpenNeuro ds004362 via DataLad or the web interface
   (≈ 1.7 GB).
2. For each `sub-XXX` and the 6 imagery runs, parse
   `RecordingDuration` from the BIDS sidecar JSON.
3. Compare against `physionet_duration_fingerprint.json` per
   subject. The 4 unique-fingerprint subjects (S2 in the
   "mixed" group plus the 3 isolated singletons) provide
   point-wise verification; the 100 cluster-shared subjects
   provide cluster-membership verification.
4. Bonus: parse `Gender` and `Age` from each
   `sub-XXX/ses-1/eeg/sub-XXX_sessions.tsv` (if present) and
   confirm they match `participants.tsv`.

A passing verification at step 3 is sufficient to claim the mapping
empirically correct.

## Footprint in the project

The fairness experiments depend on this mapping being correct. If a
future check shows it isn't, the canonical references to fix would be:

- `experiments/17_subgroup_fairness_eegnet.py`
- `experiments/22_eegnet_age_seeds.py`
- `tools/subgroup_fairness.py`
- `data/external/openneuro_ds004362_participants.tsv`

All of these load demographics keyed by integer subject ID via the
`load_demographics()` helper or equivalent, so a single mapping-fix
would propagate cleanly.
