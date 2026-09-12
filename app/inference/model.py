from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from app.config import ModelLoadConfig

LossSemantics = Literal["released", "probability"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Model inference requires the 'inference' extra") from exc
    return tf


def released_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """Return the focal loss used by the released SAR-LRA reference code.

    The historical repository implementation passes ``from_logits=True`` even
    though the model's final layer is sigmoid. This function intentionally
    preserves that implementation for reproducibility; it is not a recommendation
    for new model training.
    """
    tf = _tensorflow()

    def loss(y_true: Any, y_pred: Any) -> Any:
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
        p_t = tf.math.exp(-bce)
        return alpha * (1 - p_t) ** gamma * bce

    loss.__name__ = "released_focal_loss"
    return loss


def probability_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """Return the mathematically direct probability-space variant.

    Use only for experiments or retraining after explicitly deciding to change
    the training objective. It is not used to load or run the released weights.
    """
    tf = _tensorflow()

    def loss(y_true: Any, y_pred: Any) -> Any:
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        p_t = tf.math.exp(-bce)
        return alpha * (1 - p_t) ** gamma * bce

    loss.__name__ = "probability_focal_loss"
    return loss


# Backward-compatible name. The public package now makes the historical
# semantics explicit rather than silently substituting a different objective.
def focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    return released_focal_loss(alpha=alpha, gamma=gamma)


def build_model(config: ModelLoadConfig, *, compile_model: bool = False) -> Any:
    """Construct the orbit-specific CNN without loading or downloading weights.

    Inference does not require ``Model.compile``. Leaving the model uncompiled
    prevents training-loss implementation details from changing predictions or
    blocking weight loading across compatible TensorFlow/Keras releases.
    """
    try:
        from tensorflow.keras import Input, Model
        from tensorflow.keras.layers import (
            BatchNormalization,
            Concatenate,
            Conv2D,
            Dense,
            Dropout,
            Flatten,
            MaxPooling2D,
        )
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:
        raise RuntimeError("Model inference requires the 'inference' extra") from exc

    inputs = Input(shape=(config.patch_size, config.patch_size, config.channels))
    x = inputs
    for _ in range(2):
        x = Conv2D(config.filters_first_layer, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
    x = Conv2D(config.filters_first_layer, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    # Preserve the released architecture exactly: the third convolutional output
    # is concatenated three times in the published deployment implementation.
    x = Concatenate(axis=-1)([x, x, x])
    x = MaxPooling2D()(x)
    x = Dropout(config.dropout)(x)
    x = Flatten()(x)
    x = Dense(config.filters_first_layer * 8, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)

    if compile_model:
        model.compile(
            loss=released_focal_loss(),
            optimizer=Adam(config.learning_rate),
            metrics=["accuracy"],
        )
    return model


def compile_for_training(model: Any, config: ModelLoadConfig, *, semantics: LossSemantics) -> Any:
    """Explicitly compile a model for training under selected loss semantics."""
    try:
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:
        raise RuntimeError("Model training requires the 'inference' extra") from exc

    if semantics == "released":
        loss = released_focal_loss()
    elif semantics == "probability":
        loss = probability_focal_loss()
    else:  # pragma: no cover - protected by the Literal type for typed callers
        raise ValueError(f"Unsupported loss semantics: {semantics}")

    model.compile(loss=loss, optimizer=Adam(config.learning_rate), metrics=["accuracy"])
    return model


def load_model(config: ModelLoadConfig) -> tuple[Any, str]:
    """Build and load a local weight file without compiling or downloading it."""
    if not config.weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {config.weights_path}")
    model = build_model(config, compile_model=False)
    model.load_weights(config.weights_path)
    return model, sha256_file(config.weights_path)
