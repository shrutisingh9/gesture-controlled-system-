"""Train a simple gesture classifier with TensorFlow/Keras."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


def train(dataset_csv: str = "gesture_dataset.csv", model_path: str = "gesture_model.h5") -> None:
    data = pd.read_csv(dataset_csv)
    x = data.drop(columns=["label"]).values.astype("float32")
    y_raw = data["label"].values

    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y_raw)
    y = keras.utils.to_categorical(y_int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(x.shape[1],)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(y.shape[1], activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=0.2, verbose=1)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")
    model.save(model_path)

    labels = {i: label for i, label in enumerate(encoder.classes_)}
    np.save("gesture_labels.npy", labels, allow_pickle=True)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
