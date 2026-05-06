"""Filter-Bank Common Spatial Patterns + LDA.

The classical motor-imagery pipeline. We split the broadband EEG into N
frequency sub-bands, compute CSP within each band, take the log-variance
of the top-k CSP components per band, concatenate, and run LDA on the
result.

Original reference: Ang et al., "Filter Bank Common Spatial Pattern
(FBCSP) in Brain-Computer Interface", IJCNN 2008.

For attacks the embedding is the (concat of per-band CSP log-variances)
vector — i.e., the input to LDA. That's the semantically right level: it's
what an attacker who has access to the model's pre-classifier features
would see.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from models.base import VictimModel


def _bandpass_array(X: np.ndarray, sfreq: float, low: float, high: float,
                    order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass over a (n_windows, n_channels, n_times) array.

    We use a 4th-order Butterworth IIR via second-order sections instead of
    mne's FIR filter because FIR filters with ~4-Hz wide passbands need
    transition lengths of ~1.5 s, which is most of a 2-s motor-imagery
    window. Butterworth IIR has a much shorter impulse response and gives
    stable narrowband filtering on short windows.
    """
    nyq = sfreq / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    # filtfilt operates along axis=-1 (time) by default — exactly what we want.
    return sosfiltfilt(sos, X.astype(np.float64), axis=-1).astype(np.float32, copy=False)


@dataclass
class _Band:
    low: float
    high: float


# Standard FBCSP filter bank — 4-Hz wide overlapping bands across the mu/beta range.
DEFAULT_BANDS: tuple[_Band, ...] = (
    _Band(4, 8), _Band(8, 12), _Band(12, 16), _Band(16, 20),
    _Band(20, 24), _Band(24, 28), _Band(28, 32), _Band(32, 36), _Band(36, 40),
)


class FBCSPVictim(VictimModel):
    """Filter-Bank CSP + LDA classifier.

    Parameters
    ----------
    sfreq : sampling rate of the input windows.
    n_components : CSP filters retained per band (must be even; standard 4).
    bands : iterable of (low_hz, high_hz) sub-band tuples. Defaults to a
        9-band 4–40 Hz bank covering mu/beta.
    """
    name = "fbcsp_lda"

    def __init__(
        self,
        *,
        sfreq: float,
        n_components: int = 4,
        bands: tuple[_Band, ...] = DEFAULT_BANDS,
        n_classes: int = 4,
    ) -> None:
        self.sfreq = sfreq
        self.n_components = n_components
        self.bands = bands
        self.n_classes = n_classes
        self.csps_: list[CSP] | None = None
        self.scaler_: StandardScaler | None = None
        self.lda_: LinearDiscriminantAnalysis | None = None

    # ---- internals -------------------------------------------------------
    def _band_features(self, X: np.ndarray) -> np.ndarray:
        """For each band, run filtered windows through that band's fitted CSP
        and return the log-variance of CSP components. Concatenate over bands.
        """
        assert self.csps_ is not None
        feats = []
        for csp, band in zip(self.csps_, self.bands):
            Xb = _bandpass_array(X, self.sfreq, band.low, band.high)
            # mne CSP expects float64; transform_into='average_power' returns
            # log-variance directly.
            feats.append(csp.transform(Xb.astype(np.float64)))
        return np.concatenate(feats, axis=1).astype(np.float32, copy=False)

    # ---- VictimModel API -------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSPVictim":
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        self.csps_ = []
        feats = []
        for band in self.bands:
            Xb = _bandpass_array(X, self.sfreq, band.low, band.high)
            csp = CSP(n_components=self.n_components,
                      transform_into="average_power",
                      log=True, norm_trace=False)
            csp.fit(Xb.astype(np.float64), y)
            feats.append(csp.transform(Xb.astype(np.float64)))
            self.csps_.append(csp)
        F = np.concatenate(feats, axis=1).astype(np.float32, copy=False)

        self.scaler_ = StandardScaler().fit(F)
        F_std = self.scaler_.transform(F)
        self.lda_ = LinearDiscriminantAnalysis().fit(F_std, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = self._band_features(X)
        F_std = self.scaler_.transform(F)
        return self.lda_.predict(F_std).astype(np.int64)

    def embed(self, X: np.ndarray) -> np.ndarray:
        """The concatenated, standardized CSP log-variance features —
        i.e., LDA's input. Dimension = n_components × n_bands."""
        F = self._band_features(X)
        return self.scaler_.transform(F).astype(np.float32, copy=False)
