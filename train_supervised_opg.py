from pathlib import Path

import tensorflow as tf


def main() -> None:
    data_dir = Path("opg with landmarks") / "augmented_classwise"
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    image_size = (128, 128)
    batch_size = 32
    seed = 42

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=0.3,
        subset="training",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=0.3,
        subset="validation",
    )

    class_names = train_ds.class_names
    train_count = int(tf.data.experimental.cardinality(train_ds).numpy() * batch_size)
    val_count = int(tf.data.experimental.cardinality(val_ds).numpy() * batch_size)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255.0, input_shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
        )
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=callbacks, verbose=1)

    eval_loss, eval_acc, eval_precision, eval_recall = model.evaluate(val_ds, verbose=0)
    f1 = (2 * eval_precision * eval_recall) / (eval_precision + eval_recall + 1e-8)

    model.save(model_dir / "opg_supervised_cnn_70_30.keras")

    print(f"Classes: {class_names}")
    print(f"Approx train images (70%): {train_count}")
    print(f"Approx validation images (30%): {val_count}")
    print(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Evaluation accuracy: {eval_acc:.4f}")
    print(f"Evaluation precision: {eval_precision:.4f}")
    print(f"Evaluation recall: {eval_recall:.4f}")
    print(f"Evaluation F1: {f1:.4f}")
    print(f"Saved model: {model_dir / 'opg_supervised_cnn_70_30.keras'}")


if __name__ == "__main__":
    main()
