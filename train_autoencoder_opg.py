from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split


def load_images(image_dir: Path, image_size: tuple[int, int] = (128, 128)) -> np.ndarray:
    image_paths = sorted(
        [p for p in image_dir.glob("*.jpg")] + [p for p in image_dir.glob("*.png")]
    )
    if not image_paths:
        raise ValueError(f"No images found in: {image_dir}")

    data = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("L").resize(image_size)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        data.append(arr)

    x = np.array(data, dtype=np.float32)
    return np.expand_dims(x, axis=-1)


def build_autoencoder(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D(2, padding="same")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    encoded = tf.keras.layers.MaxPooling2D(2, padding="same")(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(encoded)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D(2)(x)
    outputs = tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(inputs, outputs, name="opg_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def main() -> None:
    input_dir = Path("opg with landmarks") / "augmented_800"
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    x = load_images(input_dir, image_size=(128, 128))

    x_train, x_val = train_test_split(x, test_size=0.2, random_state=42, shuffle=True)

    model = build_autoencoder(input_shape=(128, 128, 1))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train,
        x_train,
        validation_data=(x_val, x_val),
        epochs=20,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(model_dir / "opg_autoencoder.keras")
    np.save(model_dir / "autoencoder_train_loss.npy", np.array(history.history["loss"]))
    np.save(model_dir / "autoencoder_val_loss.npy", np.array(history.history["val_loss"]))

    print(f"Total images: {len(x)}")
    print(f"Train images (80%): {len(x_train)}")
    print(f"Validation images (20%): {len(x_val)}")
    print(f"Final train loss: {history.history['loss'][-1]:.6f}")
    print(f"Final val loss: {history.history['val_loss'][-1]:.6f}")
    print(f"Saved model: {model_dir / 'opg_autoencoder.keras'}")


if __name__ == "__main__":
    main()
