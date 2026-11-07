"""EfficientNet-B4 video classifier and loading of the released checkpoints."""

import tempfile
import zipfile
from pathlib import Path

import keras
import tensorflow as tf

from .config import CONFIGS, IMG_SIZE, SEQ_LEN

HUB_REPO = "mertkayacs/xdfdet"
RELEASED = [name for name in CONFIGS if name != "aug-intense"]  # that checkpoint was lost
WEIGHTS_FILE = "model.weights.h5"  # the weights inside every .keras archive


def build_model(dropout=0.55, seq_len=SEQ_LEN, backbone_weights="imagenet"):
    """TimeDistributed EfficientNet-B4 -> average pooling -> dropout -> sigmoid per frame.

    The output is the per-frame probability that the video is real; the video score
    is the mean over frames. Keras' EfficientNet rescales and normalizes its input
    internally, and the original pipeline also fed ImageNet-normalized frames. The
    released weights were trained that way, so `video.normalize` must stay in front.
    """
    inputs = keras.layers.Input((seq_len, IMG_SIZE, IMG_SIZE, 3))
    backbone = keras.applications.EfficientNetB4(
        weights=backbone_weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.layers.TimeDistributed(backbone)(inputs)
    x = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D())(x)
    x = keras.layers.TimeDistributed(keras.layers.Dropout(dropout))(x)
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(
        1, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(1e-3), dtype="float32"))(x)
    return keras.Model(inputs, outputs)


def video_loss(y_true, y_pred):
    """Binary cross-entropy on the frame-averaged prediction, matching the inference rule."""
    return keras.losses.binary_crossentropy(tf.reduce_mean(y_true, axis=1), tf.reduce_mean(y_pred, axis=1))


def load_weights(model, path):
    """Copy the weights of a .keras archive into `model`.

    A .keras file is a zip with config.json, metadata.json and model.weights.h5. Only the
    weights are read: the saved config is never deserialized, so nothing in the file can
    run as code, and Keras releases that cannot rebuild the saved config still work.
    """
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as tmp:
        weights = Path(tmp) / WEIGHTS_FILE
        weights.write_bytes(archive.read(WEIGHTS_FILE))
        model.load_weights(weights)
    return model


def load_model(name_or_path, mixed_precision=False):
    """Load a released configuration by name (downloaded from Hugging Face) or a local .keras file.

    The architecture comes from `build_model` and only the weights come from the file. The
    released files are float32. `mixed_precision=True` builds the model in mixed float16,
    the precision of the original training and test runs, for reproducing the reported
    scores on a GPU. Scores can differ slightly between the two.
    """
    path, dropout = name_or_path, None
    if name_or_path in CONFIGS:
        if name_or_path not in RELEASED:
            raise ValueError(f"no released checkpoint for {name_or_path!r}; train it with `xdfdet train`")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HUB_REPO, f"{name_or_path}.keras")
        dropout = CONFIGS[name_or_path].dropout
    if mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    # Dropout is inactive at inference; a local file keeps the default rate of build_model.
    model = build_model() if dropout is None else build_model(dropout=dropout)
    return load_weights(model, path)


def video_score(model, frames):
    """Mean real-probability over the frames of one normalized sequence of shape (T, H, W, 3)."""
    return float(model.predict(frames[None], verbose=0).mean())
