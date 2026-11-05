"""
splits package

Provides identity-disjoint partition and k-fold cross-validation generators
to guarantee zero identity leakage between training, validation, and evaluation splits.
"""

from splits.identity_disjoint_split import (
    generate_ctts80_kfold_splits,
    generate_digiface_disjoint_splits,
    generate_fiw_disjoint_splits,
)

__all__ = [
    "generate_ctts80_kfold_splits",
    "generate_fiw_disjoint_splits",
    "generate_digiface_disjoint_splits",
]

