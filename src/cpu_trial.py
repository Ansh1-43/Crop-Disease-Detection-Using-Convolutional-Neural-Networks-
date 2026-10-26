
from pathlib import Path
import pandas as pd
import tensorflow as tf

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
TRAIN_SAMPLES = 304
VAL_SAMPLES = 76
EPOCHS = 1
SEED = 42

# Explicitly use CPU for this trial
tf.config.set_visible_devices([], "GPU")

def load_sample(csv_name, sample_count, class_to_id, training=False):
    df = pd.read_csv(SPLIT_DIR / csv_name)

    # Sample from the existing split; never mix train/validation/test.
    df = (
        df.groupby("class_name", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(len(group), max(1, sample_count // df["class_name"].nunique())),
                random_state=SEED,
            ),
            include_groups=True,
        )
        .reset_index(drop=True)
    )

    paths = df["image_path"].astype(str).to_numpy()
    labels = df["class_name"].map(class_to_id).astype("int32").to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(len(df), seed=SEED)

    def read_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    return ds.map(read_image, num_parallel_calls=2).batch(BATCH_SIZE).prefetch(1), df


def main():
    train_csv = SPLIT_DIR / "color_train.csv"
    val_csv = SPLIT_DIR / "color_validation.csv"

    train_df = pd.read_csv(train_csv)
    classes = sorted(train_df["class_name"].unique())
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    train_ds, train_sample = load_sample(
        "color_train.csv", TRAIN_SAMPLES, class_to_id, training=True
    )
    val_ds, val_sample = load_sample(
        "color_validation.csv", VAL_SAMPLES, class_to_id
    )

    print("TensorFlow:", tf.__version__)
    print("Training device: CPU")
    print("Classes:", len(classes))
    print("Training sample:", len(train_sample))
    print("Validation sample:", len(val_sample))

    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(classes), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
    )

    output_dir = PROJECT_ROOT / "models" / "cpu_trial"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "color_trial.keras")
    print("Trial model saved to:", output_dir / "color_trial.keras")


if __name__ == "__main__":
    main()
