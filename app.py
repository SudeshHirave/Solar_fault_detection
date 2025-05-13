import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Solar Image Analyzer",
    page_icon="☀️",
    layout="centered"
)

def preprocess_image(image, target_size=(244, 244)):
    """
    Resize and normalize the input image to match model input.
    """
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = tf.keras.applications.vgg16.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)  # Shape: (1, 244, 244, 3)

@st.cache_resource
def load_model_from_file():
    """
    Load the pre-trained model from file (.keras format).
    Uses caching to prevent reloading on each rerun.
    """
    try:
        model = load_model("models/solar_keras.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def interpret_prediction(prediction):
    """
    Convert model prediction to human-readable output.
    Adjust based on your model's output.
    """
    if isinstance(prediction, np.ndarray) and len(prediction.shape) == 2:
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx] * 100
        class_names = [
    "Bird_drop_generateds",
    "Clean",
    "Dusty",
    "Electrical_damage_generated",
    "Physcial_damage_generated",
    "Snow_covered_generated"
]

        print(prediction[0])
        if class_idx < len(class_names):
            return f"Prediction: {class_names[class_idx]} with {confidence:.2f}% confidence"
        else:
            return f"Prediction: Class {class_idx} with {confidence:.2f}% confidence"
    else:
        return f"Model output: {prediction}"

def main():
    st.title("☀️ Solar Image Analyzer")

    st.markdown("""
    ## Upload an image for analysis
    This application uses a pre-trained deep learning model to analyze solar images and provide predictions.

    ### Instructions:
    1. Upload a solar-related image using the file uploader below
    2. Wait for the model to process your image
    3. View the prediction results

    Supported formats: JPG, PNG
    """)

    model = load_model_from_file()

    if model is None:
        st.error("Failed to load the model. Please check if the `.keras` file exists in the 'models/' directory.")
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.write("Image successfully uploaded. Processing...")

            with st.spinner("Analyzing image..."):
                processed_img = preprocess_image(image)
                prediction = model.predict(processed_img)
                result = interpret_prediction(prediction)

            st.success("Analysis complete!")
            st.subheader("Prediction Result:")
            st.write(result)

        except Exception as e:
            st.error(f"Error processing the image: {e}")
            st.write("Please ensure you're uploading a valid image file in JPG or PNG format.")

if __name__ == "__main__":
    main()