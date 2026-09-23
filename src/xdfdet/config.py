"""Experiment settings, taken from the original training notebooks."""

from dataclasses import dataclass

IMG_SIZE = 224
SEQ_LEN = 12                  # frames per video fed to the model
CROP_FRAMES = 32              # frames kept per video by the face-crop step
TRAIN_FRAME_STEP = 3          # on 32-frame clips this falls back to 12 evenly spaced frames
EVAL_FRAME_STEP = 2           # frames 0, 2, ..., 22: the sampling behind the reported test scores

MEAN = (0.485, 0.456, 0.406)  # ImageNet statistics
STD = (0.229, 0.224, 0.225)

# SSIM-guided polygon cutout on fake frames
SSIM_BIN = 0.5
RHO_THRESH = 0.3
MIN_AREA_RATIO = 0.02
MAX_AREA_RATIO = 0.05
P_CUTOUT = 0.5

# Star cutout on real frames
STAR_MIN_R = 8
STAR_MAX_R = 16

# Training
BATCH_PAIRS = 4
EPOCHS = 20
LEARNING_RATE = 1e-3
DECAY_STEPS = 1000
PATIENCE = 5
SEED = 42

# Albumentations probabilities per level: (noise, blur, color, geometry).
# Every level keeps HorizontalFlip at its default p=0.5, including "flip",
# which is what the cutout-only notebooks ran with.
AUGMENT_LEVELS = {
    "flip": (0.0, 0.0, 0.0, 0.0),
    "standard": (0.05, 0.05, 0.1, 0.1),
    "intense": (0.2, 0.2, 0.5, 0.5),
}


@dataclass(frozen=True)
class Config:
    augment: str | None     # key of AUGMENT_LEVELS, or None
    fill: str | None        # cutout fill: "black", "white", "random", or None
    dropout: float = 0.55


CONFIGS = {
    "baseline": Config(None, None),
    "aug-standard": Config("standard", None),
    "aug-intense": Config("intense", None),
    "cutout-random": Config("flip", "random"),
    "cutout-black": Config("flip", "black"),
    "cutout-white": Config("flip", "white"),
    # The released checkpoint for this configuration was trained with dropout 0.25.
    "aug-cutout-random": Config("standard", "random", dropout=0.25),
    "aug-cutout-black": Config("standard", "black"),
    "aug-cutout-white": Config("standard", "white"),
}
