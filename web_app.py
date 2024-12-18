import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model (make sure the model is saved and the path is correct)
@st.cache_resource
def load_trained_model():
    return load_model(r"C:\Users\reddy\OneDrive\Desktop\LUNG_DISEASE\model.h5")  # Replace with your model path

model = load_trained_model()

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    if uploaded_file is not None:  # Check if the file is uploaded
        img = Image.open(uploaded_file)
        img = img.resize((224, 224))  # Resize image to model's input size

        # Convert image to RGB if it is grayscale (1 channel)
        if img.mode != 'RGB':
            img = img.convert('RGB')  # Convert grayscale to RGB

        img_array = img_to_array(img)  # Convert the image to a NumPy array
        img_array = img_array / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Perform prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        return img, predicted_class_index
    else:
        return None, None  # Return None if no file is uploaded

# Streamlit app
st.title("Lung Disease Detection")
st.write("Upload a lung X-ray image to predict the disease.")

# Image uploader
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

# Define the classes
classes = ['Emphysema', 'Normal', 'Cardiomegaly', 'Pneumonia']

# If an image is uploaded
if uploaded_file is not None:
    # Preprocess and make prediction
    image, prediction = preprocess_image(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Get the predicted class label
    predicted_class_label = classes[prediction]

    # Display the result
    st.write(f"**Prediction:** {predicted_class_label}")
