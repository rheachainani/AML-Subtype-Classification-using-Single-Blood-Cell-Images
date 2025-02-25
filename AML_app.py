import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import joblib
from huggingface_hub import hf_hub_download
import dill

dill.settings['recurse'] = True

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder

input_dim_value = 8 
le = LabelEncoder()
le.classes_ = ["RUNX1-RUNX1T1", "CBFB-MYH11", "NPM1", "PML-RARA", "Control Group"] 

def create_nn_model():
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim_value, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(le.classes_), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load models
model_path = hf_hub_download(repo_id="rhea-chainani/aml_single_blood_cell", filename="single_blood_cell_classifier.keras")
single_cell_model = tf.keras.models.load_model(model_path, compile=False)
dill.settings['recurse'] = True
with open("voting_model.pkl", "rb") as file:
    voting_model = dill.load(file)

# Blood cell type labels
cell_types = [
    "Basophil", "Eosinophil", "Erythroblast", "IG", "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"
]

# AML subtype labels
aml_subtypes = ["RUNX1-RUNX1T1", "CBFB-MYH11", "NPM1", "PML-RARA", "Control Group"]

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (360, 363))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def classify_cells(images):
    counts = np.zeros(len(cell_types))
    for img in images:
        processed_img = preprocess_image(img)
        prediction = single_cell_model.predict(processed_img)
        cell_type_idx = np.argmax(prediction)
        counts[cell_type_idx] += 1
    return counts


def predict_aml_subtype(counts):
    normalized_counts = counts / np.sum(counts)  # Normalize counts
    subtype_prediction = voting_model.predict(np.expand_dims(normalized_counts, axis=0))
    subtype_idx = np.argmax(subtype_prediction)
    return aml_subtypes[subtype_idx]

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["About AML", "Classify Blood Cells"])
    
    if page == "About AML":
        st.title("Acute Myeloid Leukemia (AML)")
        st.write("""
        Acute Myeloid Leukemia (AML) is a fast-progressing cancer of the blood and bone marrow, primarily affecting white blood cells. Our application classifies AML into four genetically defined subtypes and a control group, each with unique clinical characteristics that guide treatment decisions:
\n
**RUNX1-RUNX1T1:** This subtype involves a chromosomal translocation that fuses the RUNX1 and RUNX1T1 genes. It typically presents in younger AML patients and is associated with favorable treatment responses. \n
**CBFB-MYH11:** Characterized by the fusion of the CBFB and MYH11 genes, this subtype is also often seen in younger patients and has a relatively positive prognosis with targeted therapies. \n
**NPM1:** A common mutation in AML, the NPM1 subtype is characterized by mutations in the NPM1 gene. While generally responsive to treatment, its prognosis can vary based on additional genetic factors. \n
**PML-RARA:** This subtype results from the fusion of the PML and RARA genes, leading to a distinct form of AML known as Acute Promyelocytic Leukemia (APL). It has a highly specific treatment protocol and, if treated promptly, can have an excellent prognosis. \n
**Control Group:** This group includes samples without AML, serving as a baseline for comparison and ensuring accuracy in the classification of AML subtypes.
        """)
    
    elif page == "Classify Blood Cells":
        st.title("AML Subtype Classification")
        uploaded_files = st.file_uploader("Upload Blood Cell Images", type=["jpg", "jpeg", "png", "tiff"], accept_multiple_files=True)
        
        if uploaded_files:
            images = [cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1) for file in uploaded_files]
            
            # Count blood cell types
            cell_counts = classify_cells(images)
            
            # Display counts
            st.write("### Count of Blood Cell Types")
            count_dict = {cell_types[i]: cell_counts[i] for i in range(len(cell_types))}
            st.json(count_dict)
            
            # Predict AML subtype
            predicted_subtype = predict_aml_subtype(cell_counts)
            st.write("### Predicted AML Subtype: ", predicted_subtype)
            
if __name__ == "__main__":
    main()