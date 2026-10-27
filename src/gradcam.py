"""
Grad-CAM for the trained MobileNetV2 color model.

Run from the project root:
    python src/gradcam.py "FULL_PATH_TO_A_TEST_LEAF_IMAGE.jpg"

Outputs:
    reports/gradcam/gradcam_overlay.png
    reports/gradcam/gradcam_heatmap.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mobilenetv2_cpu" / "color" / "best_model.keras"
CLASS_MAPPING_PATH = PROJECT_ROOT / "models" / "mobilenetv2_cpu" / "color" / "class_mapping.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "gradcam"
IMAGE_SIZE = (224, 224)


def load_class_names(path: Path, number_of_classes: int):
    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Training script typically saves {class_name: class_id}.
    if mapping and all(isinstance(v, (int, float)) for v in mapping.values()):
        names = [None] * number_of_classes
        for class_name, class_id in mapping.items():
            idx = int(class_id)
            if 0 <= idx < number_of_classes:
                names[idx] = class_name
        return [name if name is not None else f"class_{i}" for i, name in enumerate(names)]

    # Also support {"0": "class_name", ...}.
    if mapping and all(str(k).isdigit() for k in mapping.keys()):
        return [str(mapping.get(str(i), f"class_{i}")) for i in range(number_of_classes)]

    return [f"class_{i}" for i in range(number_of_classes)]


def find_mobilenet_base(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
            return layer
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and any(
            "mobilenetv2" in nested.name.lower() for nested in layer.layers
        ):
            return layer
    raise ValueError(
        "Could not find the nested MobileNetV2 base model in best_model.keras."
    )


def prepare_image(image_path: Path):
    original = Image.open(image_path).convert("RGB")
    resized = original.resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    # Matches the preprocessing used during training.
    input_array = tf.keras.applications.mobilenet_v2.preprocess_input(array.copy())
    return original, resized, np.expand_dims(input_array, axis=0)


def make_gradcam(model, image_batch, predicted_class_index):
    """
    Build a connected gradient path through the nested MobileNetV2 base.
    The feature model returns both the last convolutional activations and
    the base model output, so the classification score remains connected
    to the convolutional activations for gradient calculation.
    """
    base_model = find_mobilenet_base(model)
    base_index = model.layers.index(base_model)

    try:
        target_layer = base_model.get_layer("out_relu")
    except ValueError:
        conv_layers = [
            layer for layer in base_model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        if not conv_layers:
            raise ValueError("No Conv2D layer found in the MobileNetV2 base model.")
        target_layer = conv_layers[-1]

    # Reproduce any layers (e.g. augmentation) that occur before MobileNetV2
    # using the real input tensor, not a disconnected symbolic submodel.
    x = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    for layer in model.layers[1:base_index]:
        x = layer(x, training=False)

    feature_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[target_layer.output, base_model.output],
        name="gradcam_feature_model",
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = feature_model(x, training=False)

        # Continue through the classification head after the MobileNetV2 base.
        predictions = base_outputs
        for layer in model.layers[base_index + 1:]:
            predictions = layer(predictions, training=False)

        class_score = predictions[:, predicted_class_index]

    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise RuntimeError(
            "Gradients could not be computed for the selected convolutional layer. "
            "Please share the model.summary() output so the graph can be adapted."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    return heatmap.numpy()


def save_visualizations(resized_image, heatmap, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], IMAGE_SIZE, method="bilinear"
    ).numpy().squeeze()

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap_resized, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_heatmap.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    image_array = np.asarray(resized_image).astype(np.float32) / 255.0
    colored_heatmap = plt.get_cmap("jet")(heatmap_resized)[..., :3]
    overlay = np.clip(0.55 * image_array + 0.45 * colored_heatmap, 0, 1)

    plt.figure(figsize=(7, 7))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_overlay.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM heatmap for a leaf image.")
    parser.add_argument(
        "image_path",
        help="Full path to a leaf image, preferably from the color test split.",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path.strip('"')).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    if not CLASS_MAPPING_PATH.is_file():
        raise FileNotFoundError(f"Class mapping not found: {CLASS_MAPPING_PATH}")

    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    _, resized, image_batch = prepare_image(image_path)
    predictions = model.predict(image_batch, verbose=0)[0]
    class_index = int(np.argmax(predictions))
    class_names = load_class_names(CLASS_MAPPING_PATH, len(predictions))
    class_name = class_names[class_index]
    confidence = float(predictions[class_index]) * 100.0

    print(f"Image: {image_path}")
    print(f"Predicted class: {class_name}")
    print(f"Model score: {confidence:.2f}%")
    print("Note: this score is not a guarantee of real-world diagnostic certainty.")

    heatmap = make_gradcam(model, image_batch, class_index)
    save_visualizations(resized, heatmap, OUTPUT_DIR)

    print(f"Grad-CAM heatmap saved to: {OUTPUT_DIR / 'gradcam_heatmap.png'}")
    print(f"Grad-CAM overlay saved to: {OUTPUT_DIR / 'gradcam_overlay.png'}")


if __name__ == "__main__":
    main()
