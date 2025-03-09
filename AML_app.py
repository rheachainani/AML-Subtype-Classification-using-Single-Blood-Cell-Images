import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import dill
from huggingface_hub import hf_hub_download, login

input_dim_value = 8  

# Login to Hugging Face and load models
model_path = hf_hub_download(repo_id="rhea-chainani/aml_single_blood_cell", filename="model_v2.keras")
single_cell_model = tf.keras.models.load_model(model_path, compile=False)

with open("voting_model.pkl", "rb") as file:
    voting_model = dill.load(file)

# Blood cell type labels
cell_types = [
    "Basophil", "Eosinophil", "Erythroblast", "IG", "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"
]

# AML subtype labels
aml_subtypes = ['Control Group', 'CBFB-MYH11', 'NPM1', 'PML-RARA', 'RUNX1-RUNX1T1']

def preprocess_image(image):
    """
    Converts an image (read with OpenCV) to RGB, resizes it using bicubic interpolation,
    and adds a batch dimension as TensorFlow/Keras models are designed to process batches 
    of images, not single images (even single image input needs to have the shape 
    (1, height, width, channels) instead of just (height, width, channels)). 
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (144,144), interpolation=cv2.INTER_CUBIC)
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
    total_counts = np.sum(counts)
    if total_counts == 0:
        return "No cells detected"
    normalized_counts = counts / total_counts  # Normalize counts for voting model
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

**RUNX1-RUNX1T1:** This subtype involves a chromosomal translocation that fuses the RUNX1 and RUNX1T1 genes. It typically presents in younger AML patients and is associated with favorable treatment responses.\n
**CBFB-MYH11:** Characterized by the fusion of the CBFB and MYH11 genes, this subtype is also often seen in younger patients and has a relatively positive prognosis with targeted therapies.\n
**NPM1:** A common mutation in AML, the NPM1 subtype is characterized by mutations in the NPM1 gene. While generally responsive to treatment, its prognosis can vary based on additional genetic factors.\n
**PML-RARA:** This subtype results from the fusion of the PML and RARA genes, leading to a distinct form of AML known as Acute Promyelocytic Leukemia (APL). It has a highly specific treatment protocol and, if treated promptly, can have an excellent prognosis.\n
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
            count_dict = {cell_types[i]: int(cell_counts[i]) for i in range(len(cell_types))}
            st.json(count_dict)
            
            # Predict AML subtype
            predicted_subtype = predict_aml_subtype(cell_counts)
            if predicted_subtype == "No cells detected":
                st.error("No cells detected. Please upload valid images.")
            else:
                st.write("### Predicted AML Subtype:", predicted_subtype)
            
if __name__ == "__main__":
    main()
