# Problem Statement
## Crop Disease Detection Using Convolutional Neural Networks

### 1. Background
Crop diseases can damage leaves, reduce crop quality, and contribute to lower agricultural yields. Identifying visible disease symptoms early can help growers decide when to seek further assessment or take appropriate action. Manual identification may require experience and can be difficult when many images need to be reviewed.

### 2. Problem
This project aims to develop an image-based crop disease classification prototype that uses a Convolutional Neural Network (CNN) to classify a submitted leaf image into one of the disease or healthy categories represented in the selected dataset. The prototype will also display a Grad-CAM heatmap to illustrate image regions that influenced the model's prediction.

### 3. Who, What, Where, When, Why, and How (5W1H)
- **Who:** The intended users are learners, researchers, and people interested in preliminary crop-leaf disease screening. The prototype is not a substitute for an agricultural expert.
- **What:** Classify a leaf image into a supported crop/disease category and show a prediction confidence score and Grad-CAM visualization.
- **Where:** The initial solution will run as a local or demonstration web application using a supported image dataset.
- **When:** It may be used when a user wants to examine a leaf image for a preliminary classification.
- **Why:** A consistent image-classification workflow can demonstrate how computer vision may assist preliminary screening and how model predictions can be evaluated and explained.
- **How:** Use labeled leaf images, image preprocessing, data augmentation, transfer learning with ResNet50 or MobileNet, evaluation on a held-out test set, and a Streamlit interface with Grad-CAM.

### 4. Objectives
1. Inspect, document, and clean the supplied metadata dataset.
2. Obtain and organize appropriately labeled leaf images for model training and evaluation.
3. Perform exploratory data analysis (EDA) and produce at least three visualizations with written insights.
4. Train and evaluate a transfer-learning CNN model using an appropriate data split.
5. Implement Grad-CAM as a visual aid for interpreting predictions.
6. Build a working prototype that accepts a leaf image and displays the predicted class, confidence score, and heatmap.
7. Document results, limitations, data sources, and ethical considerations.

### 5. Success Criteria
The project will be assessed against the following measurable checks:
- The data dictionary describes every column in the submitted metadata CSV.
- The EDA notebook runs and contains at least three visualizations, each with a written insight.
- The selected model trains and can be reloaded for inference.
- Model performance is reported using appropriate metrics, including accuracy, precision, recall, F1-score, and a confusion matrix where applicable.
- The application accepts a supported image, returns a prediction, and displays a Grad-CAM heatmap.
- The project report and synopsis accurately describe the implementation and measured results.
- The project can be started using the instructions in the README.

Actual performance targets and results will be recorded after the dataset and baseline model have been evaluated. No model accuracy or field performance is claimed in this document.

### 6. Scope and Safety Note
This is an academic prototype. Predictions may be incorrect, especially for images, crops, diseases, or field conditions that differ from the training data. The system must not be presented as a definitive diagnosis or treatment recommendation. Agricultural expert validation is required before any field deployment.

### 7. Dataset Note
The project brief references the PlantVillage dataset, described in the brief as containing 54,306 images across 38 classes. The metadata CSV is not itself the image dataset. Any synthetic metadata rows must be clearly identified as synthetic and must not be represented as real image observations or as predictions from a trained model.
