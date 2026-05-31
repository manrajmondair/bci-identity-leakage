"""Common interface every victim model must implement.

Attacks operate on the model's *internal embedding* — the penultimate-layer
features for EEGNet, the CSP-projected log-variances for FBCSP, the
tangent-space-mapped covariances for the Riemannian tangent-space + logistic
regression decoder. Each victim wraps itself behind a single `embed()` method
so attack code stays generic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VictimModel(ABC):
    """Base class for the three BCI decoders we'll attack."""

    name: str = "victim"
    n_classes: int = 4

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> VictimModel:
        """Fit the task decoder on (windows, motor-imagery labels)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels for each window."""

    @abstractmethod
    def embed(self, X: np.ndarray) -> np.ndarray:
        """Return the internal embedding the attacks operate on.

        Shape: (n_windows, embedding_dim). Float32.
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean classification accuracy."""
        return float((self.predict(X) == y).mean())
