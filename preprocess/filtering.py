"""Bandpass + notch filtering on raw EEG. Standard motor-imagery preprocessing:
zero-phase FIR bandpass 4–40 Hz, no ICA (we want to test what a *typical* BCI
pipeline leaks, not a heavily-cleaned one)."""
from __future__ import annotations

import mne

from config import BANDPASS_HIGH, BANDPASS_LOW


def bandpass(raw: mne.io.BaseRaw, low: float = BANDPASS_LOW, high: float = BANDPASS_HIGH,
             *, copy: bool = True, verbose: bool = False) -> mne.io.BaseRaw:
    """Zero-phase FIR bandpass. Works in-place by default; pass copy=True to preserve input."""
    raw = raw.copy() if copy else raw
    raw.filter(l_freq=low, h_freq=high, fir_design="firwin", verbose=verbose)
    return raw


def notch(raw: mne.io.BaseRaw, freqs=(60.0,), *, copy: bool = True, verbose: bool = False) -> mne.io.BaseRaw:
    """Optional 60 Hz line-noise notch. PhysioNet was recorded in the US (60 Hz mains)."""
    raw = raw.copy() if copy else raw
    raw.notch_filter(freqs=list(freqs), verbose=verbose)
    return raw
