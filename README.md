# Acute Myeloid Leukemia (AML) Subtype Classification Using Single Blood Cell Images

## Project Overview

This project focuses on classifying subtypes of Acute Myeloid Leukemia (AML) using deep learning techniques. AML is a type of cancer that impacts the myeloid lineage of blood cells, often associated with specific genetic mutations. By leveraging single-cell blood smear images, the project aims to enhance diagnostic accuracy and support personalized treatment plans.

## Table of Contents
1. [Objectives](#objectives)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
   - [Single Cell Classification](#single-cell-classification)
   - [Data Aggregation](#data-aggregation)
   - [AML Subtype Classification](#aml-subtype-classification)
4. [Results](#results)
5. [User Interface](#user-interface)

## Objectives

- Improve AML subtype classification accuracy through single-cell image analysis.
- Experiment with multiple deep learning architectures for sensitivity in detecting AML subtypes.
- Validate the model's diagnostic accuracy and real-world applicability to assist healthcare professionals.

## Dataset

This project utilizes two main datasets:
1. **Peripheral Blood Cell Images Dataset**: Contains 17,092 high-resolution images of blood cells with various morphologies, annotated by pathologists.
2. **AML and Control Group Dataset**: Comprises 189 peripheral blood smears, divided by specific AML subtypes and a control group.

The data is preprocessed, ensuring class balance, and is used for training and testing deep learning models.

## Methodology

### Single Cell Classification

- A CNN model is trained on single-cell images to classify cell types such as neutrophils, lymphocytes, and platelets.
- Invalid image formats were handled with a custom data generator, improving data consistency for model training.

### Data Aggregation

- The CNN model classifies each cell type within patient folders, creating a data frame of cell counts per patient.
- SMOTE is applied to balance classes and improve predictive performance.

### AML Subtype Classification

- An ensemble approach (XGBoost, CatBoost, Random Forest, and Neural Network) is employed to predict AML subtypes.
- A binary classifier first identifies AML presence, and, if detected, the model further classifies the subtype.

![image](https://github.com/user-attachments/assets/0d873077-6487-4d6e-8513-fed141a9c13a)


## Results

- The CNN achieved 94% accuracy on unseen data, indicating robust performance.
- Performance metrics across AML subtypes and control groups demonstrate high precision, recall, and F1-scores.
- Correlations of the single blood cell types with the presence/absence of AML suggest that basophils may indicate control, while lymphocytes and monocytes correlate with AML subtypes.

## User Interface

The project includes a user-friendly interface that enables users to upload patient images for subtype classification, displaying cell type counts and AML predictions for diagnostic insight.
[Demo](Deployment.mp4) / [Deployment](https://drive.google.com/file/d/1nhQylDgrd9PtpQw-N2KuELWGN3T5DWgj/view?usp=sharing)
