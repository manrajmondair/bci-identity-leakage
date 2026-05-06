"""Riemannian motor-imagery decoder.

The standard pipeline:
  1. Compute a per-window covariance matrix Σ_w (estimated with OAS shrinkage
     to keep it well-conditioned on small windows).
  2. Map each Σ_w from the SPD manifold to its tangent space at the geometric
     mean. Tangent vectors are euclidean, so they go straight into a linear
     classifier.
  3. Logistic regression on the tangent vectors.

For attacks the embedding is the tangent-space vector. That's the
geometrically natural "feature" an attacker would extract.

Reference: Barachant et al., "Multiclass Brain-Computer Interface
Classification by Riemannian Geometry", IEEE TBME 2012.
"""
from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from models.base import VictimModel


class RiemannianVictim(VictimModel):
    """Covariance + tangent-space + logistic regression."""
    name = "riemann_ts_lr"

    def __init__(
        self,
        *,
        n_classes: int = 4,
        cov_estimator: str = "oas",
        C: float = 1.0,
        max_iter: int = 1000,
        seed: int = 0,
    ) -> None:
        self.n_classes = n_classes
        self.cov_estimator = cov_estimator
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.cov_: Covariances | None = None
        self.tangent_: TangentSpace | None = None
        self.clf_: LogisticRegression | None = None
        self._pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("cov", Covariances(estimator=self.cov_estimator)),
            ("ts", TangentSpace()),
            ("clf", LogisticRegression(
                C=self.C, max_iter=self.max_iter, random_state=self.seed,
            )),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiemannianVictim":
        # pyriemann expects float64 covariance inputs
        Xd = X.astype(np.float64, copy=False)
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(Xd, y)
        # Cache the fitted sub-estimators for embedding extraction
        self.cov_ = self._pipeline.named_steps["cov"]
        self.tangent_ = self._pipeline.named_steps["ts"]
        self.clf_ = self._pipeline.named_steps["clf"]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._pipeline is not None
        return self._pipeline.predict(X.astype(np.float64, copy=False)).astype(np.int64)

    def embed(self, X: np.ndarray) -> np.ndarray:
        """Tangent-space vectors. Dim = n_channels * (n_channels + 1) / 2
        (the upper-triangular of the symmetric matrices in tangent space)."""
        assert self.cov_ is not None and self.tangent_ is not None
        Xd = X.astype(np.float64, copy=False)
        cov = self.cov_.transform(Xd)
        tangent = self.tangent_.transform(cov)
        return tangent.astype(np.float32, copy=False)
