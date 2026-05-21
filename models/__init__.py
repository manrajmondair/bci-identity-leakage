"""Victim and embedding model families.

Exposes the three victim families benchmarked across every attack and
defense:

    - FBCSP+LDA           : models.fbcsp
    - Riemann tangent + LR: models.riemannian
    - EEGNet              : models.eegnet
    - Contrastive EEGNet  : models.contrastive (open-set verification)

All families implement the shared VictimModel API defined in models.base.
"""
