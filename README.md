# Crop Disease Detection Using CNNs

A deep-learning project that classifies plant-leaf images into one of 38 PlantVillage classes using MobileNetV2. The project includes dataset quality checks, dataset preparation, model training and evaluation, Grad-CAM visualizations, and a Streamlit web application.

> **Important:** This is an educational prototype, not a substitute for expert agricultural advice. Results can be unreliable on images that differ from the training data, including real-world field photos. The app does not reliably detect non-leaf images automatically.

## Features

- Dataset quality checks for the `color`, `grayscale`, and `segmented` PlantVillage sections.
- Dataset preparation and train/validation/test split files.
- MobileNetV2 training and comparison across the three image sections.
- Evaluation outputs, including metrics, classification reports, and confusion matrices.
- Grad-CAM heatmap and overlay visualization to help inspect image regions influencing a prediction.
- Streamlit interface for uploading an image and viewing predicted classes.
- Upload validation and user-facing error handling.

## Model results

| Model input | Test accuracy | Test images |
|---|---:|---:|
| Color | 94.09% | 8,066 |
| Grayscale | 90.07% | 8,066 |
| Segmented | 91.99% | 8,225 |

The color model is the current app model. These results apply to the project's PlantVillage test split and should not be interpreted as guaranteed performance on field images. Check the saved metrics and reports for full evaluation details.

## Dataset

This project uses the PlantVillage dataset, which contains 38 classes and approximately 54,000 images per image section. Obtain the dataset from its original or authorized distribution source and follow its license/terms.

Expected local layout:

```text
data/raw/plantvillage dataset/
├── color/
├── grayscale/
└── segmented/
```

The exact folder name used by this project is **`color`**, not `colour`.

Do not commit the complete raw dataset to GitHub unless you have checked the dataset license and repository-size constraints.

## Project structure

Adjust this tree to match the files you actually commit.

```text
CNN-D/
├── data/
│   ├── raw/                    # Dataset stored locally; usually not committed
│   └── processed/
│       └── splits/
├── docs/                       # Problem statement, scope, report, user guide
├── models/
│   └── mobilenetv2_cpu/
│       ├── color/
│       │   ├── best_model.keras
│       │   ├── class_mapping.json
│       │   └── ...             # Other training/evaluation outputs
│       ├── grayscale/
│       └── segmented/
├── reports/
│   ├── gradcam/
│   └── ...                     # Quality and evaluation reports
├── src/
│   ├── app.py
│   ├── dq.py
│   ├── dataset_prep.py
│   ├── cpu_trial.py
│   ├── train_mobilenetv2_cpu.py
│   └── gradcam.py
├── requirements.txt
├── .gitignore
└── README.md
```

Only list files and folders that actually exist in the repository.

## Requirements

- Python 3.11 was used for the current Windows experiments.
- Git
- Packages listed in `requirements.txt`
- The trained color model and its class mapping to run the app

## Setup and run locally

Run commands from the project root (`CNN-D`).

### 1. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make sure `requirements.txt` lists the packages actually used by your code, including Streamlit, TensorFlow, NumPy, Pillow, and any OpenCV or plotting packages required by your scripts.

### 3. Check model files

The app expects the trained color model and class mapping at the paths configured in `src/app.py`. Confirm these exist, for example:

```text
models/mobilenetv2_cpu/color/best_model.keras
models/mobilenetv2_cpu/color/class_mapping.json
```

Update this section if your app uses different paths or filenames.

### 4. Launch the application

```bash
python -m streamlit run src/app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Reproducing the workflow

Check each script's configuration before running it. The exact order and arguments may depend on your local script versions.

1. Run `src/dq.py` to inspect the dataset and review its generated reports.
2. Run `src/dataset_prep.py` to create the processed split files.
3. Run the appropriate training script in `src/` if you want to retrain the models.
4. Review the metrics, classification reports, and confusion matrices.
5. Run `src/gradcam.py` using the image/model arguments configured in that script.
6. Launch the Streamlit app using the command above.

Training can take substantial time and disk space. Check paths and output behavior before rerunning scripts.

## How to use the app

1. Start the Streamlit app.
2. Upload a supported image containing a plant leaf.
3. Review the predicted class and other displayed predictions.
4. If Grad-CAM is available, inspect its heatmap/overlay as a visualization aid.
5. Try multiple images; do not rely on one prediction for a crop-management decision.

A high model score is not proof that the diagnosis is correct or that the score is calibrated. Grad-CAM highlights regions associated with the prediction; it does not prove that the model learned valid disease features.

## Limitations

- Without a dedicated leaf/non-leaf detector, the model can assign a plant-disease class to unrelated images.
- PlantVillage images may not represent field conditions, lighting, backgrounds, camera quality, or disease severity in real use.
- Test accuracy does not guarantee real-world accuracy.
- Confidence-like scores may not be calibrated probabilities.
- Grad-CAM is an explanatory aid, not proof of causal reasoning.
- This prototype should not be the sole basis for pesticide use or other consequential agricultural decisions.

## GitHub and deployment notes

Before pushing to GitHub:

1. Check that `README.md`, `requirements.txt`, `.gitignore`, and required source files are present.
2. Exclude virtual environments, caches, raw datasets, temporary files, and private information.
3. Check model-file sizes. GitHub blocks regular Git files larger than 100 MB; large model files may need Git LFS or external model hosting.
4. Confirm that dataset and model licenses permit redistribution.
5. Clone the repository into a fresh folder and test the setup instructions before publishing.

Streamlit Community Cloud is one deployment option. The repository needs the app code and dependencies, and the deployed environment must be able to access the trained model files. If models are too large for the repository or deployment limits, use appropriate model hosting and configure the app to load them.

## Future improvements

- Add a separately trained leaf/non-leaf classifier.
- Evaluate on independent real-world field images.
- Improve weaker disease classes and assess per-class precision/recall.
- Add model versioning and experiment tracking.
- Add automated tests for upload validation and prediction handling.
- Improve uncertainty handling and crop-specific guidance.

## Acknowledgements

- PlantVillage dataset and its contributors.
- TensorFlow/Keras, MobileNetV2, Streamlit, and the open-source tools used in this project.

## License

Add a `LICENSE` file before publishing and choose a license suitable for your code and intended reuse. Confirm that the dataset and pretrained weights permit your planned use.
