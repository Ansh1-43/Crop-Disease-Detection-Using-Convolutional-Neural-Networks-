# MobileNetV2 baseline for CNN-D.
# Run from project root: python src/train_mobilenetv2.py

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "models" / "mobilenetv2"
REPORTS_DIR = PROJECT_ROOT / "reports"

SECTIONS = ["color", "grayscale", "segmented"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def load_split(section, split, class_to_id):
    csv_path = SPLIT_DIR / f"{section}_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"image_path", "class_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")

    paths = df["image_path"].astype(str).tolist()
    missing_paths = [p for p in paths if not Path(p).is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} image paths do not exist in {csv_path.name}. "
            f"Example: {missing_paths[0]}"
        )

    unknown = sorted(set(df["class_name"]) - set(class_to_id))
    if unknown:
        raise ValueError(f"Unexpected labels in {csv_path.name}: {unknown[:5]}")

    labels = df["class_name"].map(class_to_id).astype(np.int32).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if split == "train":
        ds = ds.shuffle(min(len(df), 10000), seed=SEED, reshuffle_each_iteration=True)

    def read_image(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    ds = ds.map(read_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds, df


def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.10)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for section in SECTIONS:
        print("\n" + "=" * 70)
        print(f"MobileNetV2 experiment: {section}")
        print("=" * 70)

        train_df = pd.read_csv(SPLIT_DIR / f"{section}_train.csv")
        classes = sorted(train_df["class_name"].dropna().unique().tolist())
        class_to_id = {name: idx for idx, name in enumerate(classes)}

        train_ds, train_df = load_split(section, "train", class_to_id)
        val_ds, val_df = load_split(section, "validation", class_to_id)
        test_ds, test_df = load_split(section, "test", class_to_id)

        print(f"Classes: {len(classes)}")
        print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")
        print("Loading ImageNet pretrained weights; first run may require internet.")

        model = build_model(len(classes))
        out_dir = OUTPUT_DIR / section
        out_dir.mkdir(parents=True, exist_ok=True)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(out_dir / "best_model.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.CSVLogger(str(out_dir / "training_history.csv")),
        ]

        history = model.fit(
            train_ds, validation_data=val_ds, epochs=EPOCHS,
            callbacks=callbacks, verbose=1
        )
        model.save(out_dir / "final_model.keras")

        best_path = out_dir / "best_model.keras"
        eval_model = tf.keras.models.load_model(best_path) if best_path.exists() else model
        test_loss, test_accuracy = eval_model.evaluate(test_ds, verbose=1)
        y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
        y_pred = np.argmax(eval_model.predict(test_ds, verbose=1), axis=1)

        report = classification_report(
            y_true, y_pred, labels=list(range(len(classes))),
            target_names=classes, output_dict=True, zero_division=0
        )
        pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv")
        pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))),
            index=classes, columns=classes
        ).to_csv(out_dir / "confusion_matrix.csv")
        with open(out_dir / "class_mapping.json", "w", encoding="utf-8") as f:
            json.dump({str(i): name for name, i in class_to_id.items()}, f, indent=2)

        result = {
            "section": section,
            "model": "MobileNetV2",
            "num_classes": len(classes),
            "train_images": len(train_df),
            "validation_images": len(val_df),
            "test_images": len(test_df),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "epochs_completed": len(history.history.get("loss", [])),
        }
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        results.append(result)
        print("Test metrics:", json.dumps(result, indent=2))
        tf.keras.backend.clear_session()

    summary_path = REPORTS_DIR / "mobilenetv2_section_comparison.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\nComparison summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
