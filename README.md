# Acute Myeloid Leukemia (AML) Subtype Classification Using Single Blood Cell Images

---
## Project Overview

This project focuses on classifying subtypes of Acute Myeloid Leukemia (AML) using deep learning techniques. AML is a type of cancer that impacts the myeloid lineage of blood cells, often associated with specific genetic mutations. By leveraging single-cell blood smear images, the project aims to enhance diagnostic accuracy and support personalized treatment plans.

---
## Table of Contents
1. [Objectives](#objectives)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
   - [Single Cell Classification](#single-cell-classification)
   - [Data Aggregation](#data-aggregation)
   - [AML Subtype Classification](#aml-subtype-classification)
4. [Results](#results)
5. [Deployment](#deployment)
   - [Environment Setup](#environment-setup)
   - [Running the Application](#running-the-application)
   - [Using the Application](#using-the-application)
   - [Model Details](#model-details)
   - [Notes](#notes)
   - [Troubleshooting](#troubleshooting)
   
---
## Objectives

- Improve AML subtype classification accuracy through single-cell image analysis.
- Experiment with multiple deep learning architectures for sensitivity in detecting AML subtypes.
- Validate the model's diagnostic accuracy and real-world applicability to assist healthcare professionals.
---
## Dataset

This project utilizes two main datasets:
1. **Peripheral Blood Cell Images Dataset**: Contains 17,092 high-resolution images of blood cells with various morphologies, annotated by pathologists.
2. **AML and Control Group Dataset**: Comprises 189 peripheral blood smears, divided by specific AML subtypes and a control group.

The data is preprocessed, ensuring class balance, and is used for training and testing deep learning models.

---

## Methodology

### Single Cell Classification

- A CNN model is trained on single-cell images to classify cell types such as neutrophils, lymphocytes, and platelets.
- Invalid image formats were handled with a custom data generator, improving data consistency for model training.

### Data Aggregation

- The CNN model classifies each cell type within patient folders, creating a data frame of cell counts per patient.
- SMOTE is applied to balance classes and improve predictive performance.

### AML Subtype Classification

- An ensemble approach (XGBoost, CatBoost, LightGBM, Random Forest, and Neural Network) is employed to predict AML subtypes.
- A binary classifier first identifies AML presence, and, if detected, the model further classifies the subtype.

![Screenshot 2025-03-10 002536](https://github.com/user-attachments/assets/0eec969a-f70d-4777-a77c-38cba5aedea2)


---
## Results

- The CNNs achieved 71-94% accuracy on unseen data, indicating robust performance.
- Performance metrics across AML subtypes and control groups demonstrate decent precision, recall, and F1-scores, with scope of improvement.
- Correlations of the single blood cell types with the presence/absence of AML suggest that basophils may indicate control, while lymphocytes and monocytes correlate with AML subtypes.

---

## Deployment

The project includes a user-friendly interface that enables users to upload patient images for subtype classification, displaying cell type counts and AML predictions for diagnostic insight.

The Streamlit application classifies AML subtypes using blood cell images. It consists of:

1. **Information Page** – Provides an overview of AML subtypes.
2. **Classification Page** – Accepts multiple blood cell images, identifies cell types, and predicts AML subtype.

The application leverages deep learning for single-cell classification and an ensemble model (`voting_model.pkl`) for final AML subtype prediction.

Check out the video demo [here](https://drive.google.com/file/d/1pmqf-FautlReBhRNr6v29fLoFYVruZTk/view).

And try it out for yourself [here](https://aml-subtype-classification.streamlit.app/)!

---

### Environment Setup

**Using Virtual Environment (Recommended)**

To create an isolated environment for this project:

```bash
python -m venv aml_env
source aml_env/bin/activate  # For Linux/macOS
aml_env\Scripts\activate     # For Windows
```

**Install Required Packages**

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, manually install dependencies:

```bash
pip install streamlit tensorflow numpy opencv-python joblib huggingface_hub dill scikeras scikit-learn
```

**Log into Hugging Face (required for private models or faster downloads)**
```bash
huggingface-cli login
```
After running this, enter your Hugging Face access token (which you can generate at huggingface.co/settings/tokens).


---

### **Running the Application**

**Start Streamlit App**

Run the following command in the project directory:

```bash
streamlit run AML_app.py
```

This will launch a local web application, accessible at:\
📌 `http://localhost:8501/`

---

### **Using the Application**

**Navigation**

- **"About AML" Page:** Provides background on AML subtypes and their significance.
- **"Classify Blood Cells" Page:**
  - Upload multiple blood cell images (**JPG, PNG, TIFF**).
  - The **single-cell classifier** (`single_blood_cell_classifier.keras`) predicts individual blood cell types.
  - The **counts of 8 blood cell types** are displayed.
  - The **voting classifier (**``**) predicts the final AML subtype or control.**

---

### Model Details

**Models Used**

- **Single-Cell Classifier:** `single_blood_cell_classifier.keras` (available via Hugging Face repo)
  - CNN model that classifies individual blood cells into 8 types.
- **AML Subtype Classifier:** `voting_model.pkl`
  - A trained ensemble model that predicts AML subtype based on aggregated blood cell counts.

---
### Notes

1. Ensure all required model files (**``** and **``**) are available in the project directory.
2. For missing models, the single-cell classifier is downloaded from Hugging Face.
3. If running on a different machine, re-install dependencies and check TensorFlow compatibility.

---
### Troubleshooting

**Model Not Found?**

- Check if the Hugging Face model download was successful.
- Verify the model files exist in the correct directory.

 **Streamlit Not Found?**

If Streamlit isn't installed, run:

```bash
pip install streamlit
```

**App Not Running?**

Try restarting the environment and re-running:

```bash
source aml_env/bin/activate  # (Linux/macOS)  
aml_env\Scripts\activate     # (Windows)  
streamlit run AML_app.py
```



