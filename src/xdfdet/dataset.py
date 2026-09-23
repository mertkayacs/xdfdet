"""Keras data pipeline that turns real/fake video pairs into training batches."""

import random

import numpy as np
import tensorflow as tf

from . import augment
from .config import BATCH_PAIRS, P_CUTOUT, RHO_THRESH, SEED, SEQ_LEN, TRAIN_FRAME_STEP, Config
from .landmarks import landmarks
from .video import normalize, read_pair

REAL, FAKE = 1.0, 0.0  # the model outputs the probability that a video is real


class PairSequence(tf.keras.utils.Sequence):
    """Batches of `batch_pairs` video pairs.

    With no augmentation and no cutout each pair gives two samples (real, fake).
    Otherwise it gives four: the two originals plus a perturbed copy of each. The
    real copy may get a star cutout, the fake copy an SSIM-guided polygon cutout
    (each with probability 0.5), and both get the configuration's augmentations.
    """

    def __init__(self, pairs, config=Config(None, None), batch_pairs=BATCH_PAIRS,
                 shuffle=True, step=TRAIN_FRAME_STEP, seed=SEED, **kwargs):
        super().__init__(**kwargs)
        self.pairs, self.config, self.batch_pairs = list(pairs), config, batch_pairs
        self.shuffle, self.step = shuffle, step
        random.seed(seed), np.random.seed(seed), tf.random.set_seed(seed)
        self.order = np.arange(len(self.pairs))
        self.on_epoch_end()

    @property
    def perturbed(self):
        return self.config.augment is not None or self.config.fill is not None

    def __len__(self):
        return len(self.pairs) // self.batch_pairs

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)

    def __getitem__(self, index):
        chosen = self.order[index * self.batch_pairs:(index + 1) * self.batch_pairs]
        samples = [s for k in chosen for s in self._samples(*self.pairs[k])]
        x = np.stack([s[0] for s in samples])
        y = np.stack([np.full((SEQ_LEN, 1), s[1], np.float32) for s in samples])
        return x, y

    def _samples(self, real_path, fake_path):
        real, fake = read_pair(real_path, fake_path, step=self.step)
        out = [(self._stack(real), REAL), (self._stack(fake), FAKE)]
        if not self.perturbed:
            return out
        real_cut, fake_cut = real, fake
        if self.config.fill is not None:
            if np.random.rand() < P_CUTOUT:
                real_cut = [augment.cut_polygon(f, augment.star_polygon(*f.shape[:2]), self.config.fill)
                            for f in real]
            if np.random.rand() < P_CUTOUT:
                polygon = self._polygon(real, fake)
                if polygon is not None:
                    fake_cut = [augment.cut_polygon(f, polygon, self.config.fill) for f in fake]
        aug = augment.augmentations(self.config.augment) if self.config.augment else None
        out += [(self._stack(real_cut, aug), REAL), (self._stack(fake_cut, aug), FAKE)]
        return out

    @staticmethod
    def _polygon(real, fake):
        """Landmarks come from the real middle frame; the cut goes on every fake frame."""
        mid = len(real) // 2
        points = landmarks(real[mid])
        if points is None:
            return None
        diff = augment.ssim_difference(real[mid], fake[mid])
        return augment.select_polygon(augment.candidate_polygons(points), diff, min_overlap=RHO_THRESH)

    @staticmethod
    def _stack(frames, aug=None):
        if aug is not None:
            frames = [aug(image=f)["image"] for f in frames]
        return np.stack([normalize(f) for f in frames])
