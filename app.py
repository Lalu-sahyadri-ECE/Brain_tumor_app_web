import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load your trained model
model = load_model("brain_tumor_classifier.keras")

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Set title
st.title("🧠 Brain Tumor MRI Classification App")
st.write("Upload an MRI image and I will try to predict the tumor type.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the image
    st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)

    # Save the uploaded image to a temp file
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess image
    img = image.load_img("temp.jpg", target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Make it a batch of 1
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Show result
    st.success(f"Prediction: {predicted_class}")
