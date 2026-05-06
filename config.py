"""Project-wide constants. Imported by everything; no heavy deps."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

for d in (CACHE_DIR, FIGURES_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)

# PhysioNet EEG-MMIDB run indexing (1-based as the dataset numbers them):
#   R01: baseline eyes open
#   R02: baseline eyes closed
#   R03, R07, R11: motor execution — open/close left/right fist
#   R04, R08, R12: motor imagery — imagine open/close left/right fist
#   R05, R09, R13: motor execution — both fists / both feet
#   R06, R10, R14: motor imagery — imagine both fists / both feet
PHYSIONET_RUNS_IMAGERY = (4, 8, 12, 6, 10, 14)
PHYSIONET_RUNS_EXECUTION = (3, 7, 11, 5, 9, 13)
PHYSIONET_RUNS_BASELINE = (1, 2)

# Subjects with known issues — recording errors, missing events. Drop everywhere.
PHYSIONET_DROP_SUBJECTS = (88, 89, 92, 100, 104)
PHYSIONET_N_SUBJECTS = 109  # before drops; n_used = 104

# Standard motor-imagery preprocessing
SAMPLING_RATE = 160  # Hz, native PhysioNet rate
BANDPASS_LOW = 8.0
BANDPASS_HIGH = 30.0
EPOCH_TMIN = 0.0   # s relative to event
EPOCH_TMAX = 4.0
WINDOW_SECONDS = 2.0
WINDOW_STRIDE_SECONDS = 1.0  # 50% overlap

# A4 open-set verification — held-out subject split
OPEN_SET_TRAIN_FRAC = 0.75  # 78 of 104 subjects for embedding training; 26 held out for verification

# A5 membership inference — shadow-model count
MI_N_SHADOWS = 20
MI_SHADOW_SUBJECT_FRAC = 0.5  # each shadow trains on 50% of available subjects

DEFAULT_SEED = 0
