import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("model_3.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    predictions = model.predict(input_arr)
    return 1 if predictions >= 0.5 else 0  # Return class based on threshold

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("MELANOMA RECOGNITION MODEL")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Joe's Melanoma DLM
    
    Project goal is to be able to take an input image of a suspicious skin lesion and diagnose via DLM if the lesion is malignant or benign. 

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a suspected mole.
    2. **Analysis:** Model will process the image using advanced algorithms to identify cancer.
    3. **Results:** View the diagnosis from the model.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image.

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 14K RGB images of benign and malignant skin lesions. The total dataset is divided into an 70/15/15 ratio of training, validation and test set preserving the directory structure.
    #### Content
    1. train (9,702 images)
    2. test (2,088 images)
    3. validation (2,089 images)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image") and test_image:
        img = Image.open(test_image)
        st.image(img, width=4, use_column_width=True)
    
    # Predict button
    if st.button("Predict") and test_image:
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Define class names for binary classification
        class_name = ['Benign', 'Malignant']
        
        st.success(f"Model is Predicting it's a {class_name[result_index]}")