"""Train a simple gesture classifier with TensorFlow/Keras."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


def train(dataset_csv: str = "gesture_dataset.csv", model_path: str = "gesture_model.h5") -> None:
    data = pd.read_csv(dataset_csv)
    x = data.drop(columns=["label", "lighting"], errors="ignore").values.astype("float32")
    y_raw = data["label"].values

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, size=x.shape).astype("float32")
    scale = rng.uniform(0.9, 1.1, size=(x.shape[0], 1)).astype("float32")
    x_aug = (x * scale) + noise
    x = np.vstack([x, x_aug]).astype("float32")
    y_raw = np.concatenate([y_raw, y_raw])

    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y_raw)
    y = keras.utils.to_categorical(y_int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(x.shape[1],)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.35),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(y.shape[1], activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1),
    )
    class_weights = {i: w for i, w in enumerate(class_weights_arr)}
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    model.fit(
        x_train,
        y_train,
        epochs=80,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")
    model.save(model_path)

    labels = {i: label for i, label in enumerate(encoder.classes_)}
    np.save("gesture_labels.npy", labels, allow_pickle=True)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
