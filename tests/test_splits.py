import numpy as np

from splits import grouped_split, original_id


def test_original_id_strips_every_augmentation_suffix():
    assert original_id("female/04.png") == "04"
    assert original_id("female/04_aug_0005.png") == "04"
    assert original_id("male/12_aug_0001_aug_0038.png") == "12"


def _dataset(n_originals=60, copies=9):
    paths, labels = [], []
    for i in range(n_originals):
        cls = i % 2
        name = "female" if cls == 0 else "male"
        paths.append(f"{name}/{i:02d}.png"); labels.append(cls)
        for k in range(copies - 1):
            paths.append(f"{name}/{i:02d}_aug_{k:04d}.png"); labels.append(cls)
    return paths, labels


def test_no_source_xray_appears_in_two_splits():
    paths, labels = _dataset()
    train, val, test, groups = grouped_split(paths, labels, seed=0)
    g = lambda ix: set(groups[ix])  # noqa: E731
    assert not g(train) & g(test)
    assert not g(train) & g(val)
    assert not g(val) & g(test)
    assert len(train) + len(val) + len(test) == len(paths)


def test_both_classes_in_every_split():
    paths, labels = _dataset()
    labels = np.asarray(labels)
    for ix in grouped_split(paths, labels, seed=1)[:3]:
        assert set(labels[ix]) == {0, 1}
