"""Training one configuration, with the settings of the original runs."""

import keras
import tensorflow as tf

from .config import CONFIGS, DECAY_STEPS, EPOCHS, LEARNING_RATE, PATIENCE, Config
from .dataset import PairSequence
from .model import build_model, video_loss


def train(name, train_pairs, val_pairs, out_path, epochs=EPOCHS):
    """Train configuration `name` and save the best weights (lowest val loss) to `out_path`."""
    config = CONFIGS[name]
    if tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")
    model = build_model(dropout=config.dropout)
    schedule = keras.optimizers.schedules.CosineDecay(LEARNING_RATE, DECAY_STEPS, alpha=0.0)
    model.compile(optimizer=keras.optimizers.Adam(schedule, clipvalue=1.0), loss=video_loss,
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, min_delta=0.001,
                                         mode="min", restore_best_weights=True, verbose=1)
    history = model.fit(PairSequence(train_pairs, config),
                        validation_data=PairSequence(val_pairs, Config(None, None), shuffle=False),
                        epochs=epochs, callbacks=[stop], verbose=1)
    keras.mixed_precision.set_global_policy("float32")
    release = build_model(dropout=config.dropout)  # same architecture, float32
    release.set_weights(model.get_weights())
    release.save(out_path)
    return history
