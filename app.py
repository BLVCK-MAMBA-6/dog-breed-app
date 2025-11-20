import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import json
import warnings
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# --- Constants from your Notebook ---
IMG_SIZE = 224
NUM_CLASSES = 120
# THIS IS THE FIX. This module outputs 1001 features, matching your .h5 file.
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5"
# Using your new weights file name
WEIGHTS_FILE = "20251112-08421762936925_full-image-set-mobilenetv2-Adam.h5" 
CLASSES_FILE = "class_names.json"

# --- 1. Model Definition (MATCHING THE SAVED FILE) ---
def create_model(module_handle, num_classes):
    """
    Creates the model architecture to EXACTLY MATCH the saved weights.
    The errors tell us there are 3 Dense layers, likely named
    dense_16, dense_17, and dense_18.
    """
    feature_extractor_layer = hub.KerasLayer(
        module_handle,
        trainable=False,
        name="feature_extraction_layer"
    )
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        feature_extractor_layer,
        # We restore the 3-layer architecture AND use the names from the errors
        tf.keras.layers.Dense(128, activation='relu', name='dense_16'), 
        tf.keras.layers.Dense(64, activation='relu', name='dense_17'),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_18")
    ])
    
    return model

# --- 2. Load Model and Class Names (The RELIABLE Way) ---
@st.cache_resource
def load_model_and_classes():
    """
    Creates the model architecture AND loads the weights into it.
    This is more robust than tf.keras.models.load_model().
    """
    try:
        # Create the model architecture
        model = create_model(MODULE_HANDLE, NUM_CLASSES)
        
        # Load the saved weights BY NAME
        # This is the critical fix
        model.load_weights(WEIGHTS_FILE, by_name=True)
        
        st.success("Model created and weights loaded successfully!")

        # Load the class names
        with open(CLASSES_FILE, 'r') as f:
            class_names = json.load(f)
            
        if len(class_names) != NUM_CLASSES:
            st.error(f"Error: Loaded {len(class_names)} class names, but model expects {NUM_CLASSES}.")
            return None, None
            
        return model, class_names
        
    except FileNotFoundError as e:
        st.error(f"Error: Missing file. Make sure '{e.filename}' is in the same folder as app.py.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.write("Full error:", str(e))
        return None, None

model, class_names = load_model_and_classes()

# --- 3. Image Preprocessing (Using your new robust version) ---
def preprocess_image(image_pil):
    """
    Prepares the uploaded PIL image for the model.
    """
    # Resize to the same size as training
    image = image_pil.resize((IMG_SIZE, IMG_SIZE)) 
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Ensure it's RGB (in case of RGBA or grayscale)
    if len(image_array.shape) == 2:  # Grayscale
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = image_array[:, :, :3]
    
    # Normalize pixel values to [0, 1]
    image_array = image_array.astype(np.float32) / 255.0
    
    # Add a "batch" dimension
    preprocessed_image = np.expand_dims(image_array, axis=0)
    
    return preprocessed_image

# --- 4. Streamlit App UI (Using your new version) ---
st.set_page_config(page_title="Dog Breed Identifier", layout="centered")
st.title("🐶 Dog Breed Identifier")

if model is None or class_names is None:
    st.warning("Model or class names could not be loaded. Please check the files and refresh.")
else:
    st.write("Upload an image of a dog, and I'll try to guess its breed!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg","png"])

    if uploaded_file is not None:
        try:
            # 1. Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='You uploaded this image:', use_column_width=True)
            
            # 2. Show a "loading" spinner while processing
            with st.spinner('Classifying...'):
                
                # 3. Preprocess the image
                preprocessed_image = preprocess_image(image)
                
                # 4. Make a prediction
                predictions = model.predict(preprocessed_image, verbose=0)
                
                # 5. Get the top prediction
                top_prediction_index = np.argmax(predictions)
                top_confidence = predictions[0][top_prediction_index]
                top_breed = class_names[top_prediction_index]
            
            # 6. Display the result
            st.success(f"**I'm {top_confidence*100:.2f}% sure this is a:**")
            
            # Clean up the name
            display_name = top_breed.replace('_', ' ').title()
            st.header(f"{display_name}")
            
            # Show top 5 predictions
            with st.expander("See top 5 predictions"):
                top_5_indices = np.argsort(predictions[0])[-5:][::-1]
                for idx in top_5_indices:
                    breed_name = class_names[idx].replace('_', ' ').title()
                    confidence = predictions[0][idx] * 100
                    st.write(f"**{breed_name}**: {confidence:.2f}%")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")