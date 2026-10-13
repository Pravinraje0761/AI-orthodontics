"""Patient-safe data splitting for the OPG gender classifier.

The training folder holds several pre-augmented copies of each original X-ray
(`04.png`, `04_aug_0005.png`, `04_aug_0005_aug_0038.png`, ...). Splitting those
images at random puts copies of the same X-ray in both train and test, so test
accuracy measures recognition of seen images rather than generalisation. Every
split here is made at the level of the original X-ray instead.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

_AUG_SUFFIX = re.compile(r"(_aug_\d+)+$")


def original_id(path: str | Path) -> str:
    """Identifier of the source X-ray an image was derived from.

    >>> original_id("female/04_aug_0005_aug_0038.png")
    '04'
    """
    return _AUG_SUFFIX.sub("", Path(path).stem)


def grouped_split(paths: list[str], labels: list[int], seed: int = 42,
                  test_fraction: float = 0.2, val_fraction: float = 0.2):
    """Split indices into train / validation / test with no source X-ray shared.

    Each fraction is approximate: whole groups move together, stratified by class.
    `val_fraction` is taken from what remains after the test split.
    """
    labels = np.asarray(labels)
    # Group by class as well as id, so "04" in female/ and male/ stay distinct.
    groups = np.array([f"{y}/{original_id(p)}" for p, y in zip(paths, labels)])
    idx = np.arange(len(labels))

    def _one_fold(sub_idx, fraction, rng_seed):
        n_splits = max(2, round(1 / fraction))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
        keep, hold = next(cv.split(sub_idx, labels[sub_idx], groups[sub_idx]))
        return sub_idx[keep], sub_idx[hold]

    trainval, test = _one_fold(idx, test_fraction, seed)
    train, val = _one_fold(trainval, val_fraction, seed + 1)
    return train, val, test, groups
