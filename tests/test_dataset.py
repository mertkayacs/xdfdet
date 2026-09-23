import numpy as np

from xdfdet.config import CONFIGS
from xdfdet.data import pair_videos
from xdfdet.dataset import PairSequence


def test_baseline_batches_hold_originals_only(clip_tree):
    seq = PairSequence(pair_videos(clip_tree), CONFIGS["baseline"], batch_pairs=2)
    x, y = seq[0]
    assert x.shape == (4, 12, 224, 224, 3) and y.shape == (4, 12, 1)
    assert y[:, 0, 0].tolist() == [1, 0, 1, 0]
    assert len(seq) == 4


def test_perturbed_batches_add_a_copy_of_each(clip_tree):
    seq = PairSequence(pair_videos(clip_tree), CONFIGS["aug-cutout-black"], batch_pairs=2)
    x, y = seq[0]
    assert x.shape[0] == 8 and y[:, 0, 0].tolist() == [1, 0, 1, 0] * 2
    assert np.isfinite(x).all()
