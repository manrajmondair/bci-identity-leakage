"""Ad-hoc privacy transformations applied to the raw EEG channel dimension.

PCA channel compression, additive Gaussian noise on channels, and
channel-subset reduction. They serve as defense baselines against the
more principled DANN and DP-SGD methods.

All transforms operate on the raw windowed EEG (n_windows, n_channels,
n_times). Defender fits the transform on training data, applies it to
both train and test before the victim sees it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class ChannelPCA:
    """Channel-mode PCA: reduces the n_channels dimension to top-k principal
    components fitted across (window, time) observations of the training set.

    Output shape: (n_windows, k, n_times). The victims' n_channels parameter
    must be set to k.
    """
    k: int
    _pca: PCA | None = None
    _mean: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> ChannelPCA:
        n, c, t = X.shape
        flat = X.transpose(0, 2, 1).reshape(n * t, c).astype(np.float64, copy=False)
        self._mean = flat.mean(axis=0)
        self._pca = PCA(n_components=self.k, svd_solver="auto", random_state=0)
        self._pca.fit(flat - self._mean)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._pca is not None and self._mean is not None
        n, c, t = X.shape
        flat = X.transpose(0, 2, 1).reshape(n * t, c).astype(np.float64, copy=False)
        proj = self._pca.transform(flat - self._mean)        # (n*t, k)
        out = proj.reshape(n, t, -1).transpose(0, 2, 1)       # (n, k, t)
        return out.astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class ChannelGaussianNoise:
    """Additive zero-mean Gaussian noise scaled by channel std on the train set.

    sigma = 0.0  -> no defense; sigma >= 1 -> noise std equal to signal std.
    Same shape in/out (n_windows, n_channels, n_times).
    """
    sigma: float
    seed: int = 0
    _channel_std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> ChannelGaussianNoise:
        # Per-channel std across (window, time) on the training set
        self._channel_std = X.std(axis=(0, 2)).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._channel_std is not None
        rng = np.random.default_rng(self.seed)
        noise = rng.standard_normal(size=X.shape).astype(np.float32)
        scale = (self.sigma * self._channel_std)[None, :, None]
        return X + noise * scale

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class ChannelDrop:
    """Keep only the top-k highest-variance channels of the training set.

    Output shape: (n_windows, k, n_times).
    """
    k: int
    _kept: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> ChannelDrop:
        var = X.var(axis=(0, 2))
        # Indices of top-k by variance
        self._kept = np.argsort(-var)[: self.k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._kept is not None
        return X[:, self._kept, :].astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
