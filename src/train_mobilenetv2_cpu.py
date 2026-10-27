from pathlib import Path
import json, os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODEL_ROOT = PROJECT_ROOT / "models" / "mobilenetv2_cpu"
REPORTS_DIR = PROJECT_ROOT / "reports"
SECTIONS = ["color", "grayscale", "segmented"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42
MAP_PARALLEL_CALLS = 2
PREFETCH_BATCHES = 1


def configure_cpu():
    tf.config.set_visible_devices([], "GPU")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(max(1, (os.cpu_count() or 4) // 2))
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass


def read_split_csv(section, split):
    path = SPLIT_DIR / f"{section}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    df = pd.read_csv(path)
    required = {"image_path", "class_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path.name} contains no rows.")
    return df


def make_dataset(df, class_to_id, training):
    paths = df["image_path"].astype(str).to_numpy()
    labels = df["class_name"].map(class_to_id)
    if labels.isna().any():
        bad = sorted(df.loc[labels.isna(), "class_name"].astype(str).unique())
        raise ValueError(f"Unknown class labels: {bad}")
    labels = labels.astype(np.int32).to_numpy()

    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image paths missing. Example: {missing[0]}")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(df), 10000), seed=SEED, reshuffle_each_iteration=True)

    def load_image(path, label):
        encoded = tf.io.read_file(path)
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    ds = ds.map(load_image, num_parallel_calls=MAP_PARALLEL_CALLS)
    return ds.batch(BATCH_SIZE).prefetch(PREFETCH_BATCHES)


def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED)(inputs)
    x = tf.keras.layers.RandomRotation(0.04, seed=SEED)(x)
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


def run_experiment(section):
    print("\n" + "=" * 76)
    print(f"STARTING SECTION: {section}")
    print("=" * 76)
    train_df = read_split_csv(section, "train")
    val_df = read_split_csv(section, "validation")
    test_df = read_split_csv(section, "test")
    classes = sorted(train_df["class_name"].dropna().astype(str).unique().tolist())
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    for split_name, df in [("validation", val_df), ("test", test_df)]:
        unexpected = sorted(set(df["class_name"].astype(str)) - set(classes))
        if unexpected:
            raise ValueError(f"{section}_{split_name}.csv has unknown classes: {unexpected}")

    print(f"Classes: {len(classes)}")
    print(f"Images — train: {len(train_df):,} | validation: {len(val_df):,} | test: {len(test_df):,}")
    print("Checking image paths and preparing data pipelines...")
    train_ds = make_dataset(train_df, class_to_id, training=True)
    val_ds = make_dataset(val_df, class_to_id, training=False)
    test_ds = make_dataset(test_df, class_to_id, training=False)

    output_dir = MODEL_ROOT / section
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(len(classes))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_history.csv")),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1
    )
    model.save(output_dir / "final_model.keras")
    best_path = output_dir / "best_model.keras"
    eval_model = tf.keras.models.load_model(best_path) if best_path.exists() else model

    print(f"Evaluating {section} on its untouched test split...")
    test_loss, test_accuracy = eval_model.evaluate(test_ds, verbose=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    y_pred = np.argmax(eval_model.predict(test_ds, verbose=1), axis=1)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(classes))),
        target_names=classes, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))),
        index=classes, columns=classes
    ).to_csv(output_dir / "confusion_matrix.csv")
    with open(output_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({str(idx): name for name, idx in class_to_id.items()}, f, indent=2)

    result = {
        "section": section, "model": "MobileNetV2", "training_device": "CPU",
        "num_classes": len(classes), "train_images": len(train_df),
        "validation_images": len(val_df), "test_images": len(test_df),
        "test_loss": float(test_loss), "test_accuracy": float(test_accuracy),
        "epochs_completed": len(history.history.get("loss", [])),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Completed experiment:")
    print(json.dumps(result, indent=2))
    tf.keras.backend.clear_session()
    return result


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    configure_cpu()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("TensorFlow:", tf.__version__)
    print("Training device: CPU")
    print("Sections:", ", ".join(SECTIONS))
    print(f"Image size: 224x224 | batch size: {BATCH_SIZE} | max epochs: {EPOCHS}")
    print("CPU training for all three sections may take a long time.")
    results = []
    for section in SECTIONS:
        result = run_experiment(section)
        results.append(result)
        pd.DataFrame(results).to_csv(
            REPORTS_DIR / "mobilenetv2_cpu_section_comparison.csv", index=False
        )
    print("\nAll experiments completed.")
    print("Comparison report:", REPORTS_DIR / "mobilenetv2_cpu_section_comparison.csv")


if __name__ == "__main__":
    main()
