"""EfficientNet-B4 video classifier and loading of the released checkpoints."""

import keras
import tensorflow as tf

from .config import CONFIGS, IMG_SIZE, SEQ_LEN

HUB_REPO = "mertkayacs/xdfdet"
RELEASED = [name for name in CONFIGS if name != "aug-intense"]  # that checkpoint was lost


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


def load_model(name_or_path, mixed_precision=False):
    """Load a released configuration by name (downloaded from Hugging Face) or a local .keras file.

    The released files are float32. `mixed_precision=True` rebuilds the model in mixed
    float16, the precision of the original training and test runs, for reproducing the
    reported scores on a GPU. Scores can differ slightly between the two.
    """
    path = name_or_path
    if name_or_path in CONFIGS:
        if name_or_path not in RELEASED:
            raise ValueError(f"no released checkpoint for {name_or_path!r}; train it with `xdfdet train`")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HUB_REPO, f"{name_or_path}.keras")
    model = keras.saving.load_model(path, compile=False)
    if not mixed_precision:
        return model
    dropout = next(l.layer.rate for l in model.layers if isinstance(getattr(l, "layer", None), keras.layers.Dropout))
    keras.mixed_precision.set_global_policy("mixed_float16")
    mixed = build_model(dropout=dropout)
    mixed.set_weights(model.get_weights())
    return mixed


def video_score(model, frames):
    """Mean real-probability over the frames of one normalized sequence of shape (T, H, W, 3)."""
    return float(model.predict(frames[None], verbose=0).mean())
