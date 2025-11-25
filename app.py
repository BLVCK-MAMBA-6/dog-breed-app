import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import json
import warnings
import os
import google.generativeai as genai

# --- Page Config ---
st.set_page_config(
    page_title="Doggo Identifier",
    page_icon="🐾",
    layout="centered"
)

# --- Suppress Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# --- Constants ---
IMG_SIZE = 224
NUM_CLASSES = 120
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5"
WEIGHTS_FILE = "20251112-08421762936925_full-image-set-mobilenetv2-Adam.h5"
CLASSES_FILE = "class_names.json"
CONFIDENCE_THRESHOLD = 0.30  # Reject predictions below 30%

# --- Breed Corrections ---
# Map confusing/old labels to modern names
BREED_OVERRIDES = {
    "eskimo_dog": "American Eskimo / Husky Mix",
    "blenheim_spaniel": "Cavalier King Charles Spaniel",
    "boston_bull": "Boston Terrier"
}

# --- 1. Gemini Fun Fact Generator ---
def generate_fun_fact(breed):
    """Generates a fun fact using the app's shared Gemini API key."""
    try:
        # Retrieve the key from Streamlit Secrets
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
            
            # Updated to use the requested model
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"Give me one short, fun, and cute fact about the {breed} dog breed. Keep it under 30 words. Use emojis!"
            
            response = model.generate_content(prompt)
            return response.text
        else:
            return f"The {breed} is a good boy/girl! (API Key missing for fresh facts!)"
    except Exception as e:
        return f"The {breed} is a good boy/girl! (We couldn't fetch a new fact right now, but they are amazing!)"

# --- 2. Custom CSS ---
def add_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;700&display=swap');
        
        .stApp {
            background-color: #FFF5F7;
            background-image: radial-gradient(#FFE4E1 1px, transparent 1px);
            background-size: 20px 20px;
            font-family: 'Nunito', sans-serif;
        }
        
        h1 { 
            font-family: 'Fredoka One', cursive; 
            color: #FF6B6B; 
            text-align: center; 
            text-shadow: 2px 2px 0px #FFE66D; 
            font-size: 3rem;
        }
        
        h2 { color: #4ECDC4; font-family: 'Fredoka One', cursive; }
        
        /* Card Styling */
        .result-card {
            background-color: white;
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            text-align: center;
            border: 4px solid #FFD93D;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        /* Fun Fact Box */
        .fun-fact-box {
            background-color: #E0F7FA;
            padding: 20px;
            border-radius: 15px;
            border-left: 8px solid #4ECDC4;
            margin-top: 15px;
            font-style: italic;
            font-size: 1.1rem;
            color: #006064;
        }
        
        .stFileUploader { text-align: center; }
        img { border-radius: 15px; border: 4px solid #FF6B6B; }
        
        /* Button Styling */
        .stButton button {
            background-color: #FF6B6B;
            color: white;
            border-radius: 10px;
            font-family: 'Fredoka One', cursive;
        }
        </style>
    """, unsafe_allow_html=True)

add_custom_css()

# --- 3. Model Definition ---
def create_model(module_handle, num_classes):
    feature_extractor_layer = hub.KerasLayer(module_handle, trainable=False, name="feature_extraction_layer")
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        feature_extractor_layer,
        tf.keras.layers.Dense(128, activation='relu', name='dense_16'), 
        tf.keras.layers.Dense(64, activation='relu', name='dense_17'),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_18")
    ])
    return model

# --- 4. Load Model ---
@st.cache_resource
def load_model_and_classes():
    try:
        model = create_model(MODULE_HANDLE, NUM_CLASSES)
        model.load_weights(WEIGHTS_FILE, by_name=True)
        with open(CLASSES_FILE, 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

model, class_names = load_model_and_classes()

# --- 5. Preprocessing ---
def preprocess_image(image_pil):
    image = image_pil.resize((IMG_SIZE, IMG_SIZE)) 
    image_array = np.array(image)
    # Handle grayscale or RGBA
    if len(image_array.shape) == 2: image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4: image_array = image_array[:, :, :3]
    
    image_array = image_array.astype(np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

# --- 6. Main App ---
st.title("🐾 Dog Breed Identifier 🐾")
st.markdown("<h4 style='text-align: center; color: #888;'>Upload a photo and I'll guess the breed! 📸</h4>", unsafe_allow_html=True)

if model is None or class_names is None:
    st.error("⚠️ Could not load model. Please check files.")
else:
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Center the image column
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with st.spinner('🤔 Sniffing the image...'):
                preprocessed_image = preprocess_image(image)
                predictions = model.predict(preprocessed_image, verbose=0)
                
                top_prediction_index = np.argmax(predictions)
                top_confidence = predictions[0][top_prediction_index]
                raw_breed = class_names[top_prediction_index]
                
                # --- LOGIC: Is this actually a dog? ---
                if top_confidence < CONFIDENCE_THRESHOLD:
                    st.warning(f"🤔 Hmm... I'm only **{top_confidence*100:.1f}%** sure about this.")
                    st.info("This might not be a dog, or it's a breed I haven't learned yet! Try a clearer photo.")
                else:
                    # Apply Manual Overrides (Fixing Husky/Eskimo confusion)
                    if raw_breed in BREED_OVERRIDES:
                        display_name = BREED_OVERRIDES[raw_breed]
                    else:
                        display_name = raw_breed.replace('_', ' ').title()

                    # Generate Fun Fact
                    fun_fact = generate_fun_fact(display_name)

                    st.balloons()
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h2>I'm {top_confidence*100:.0f}% sure it's a...</h2>
                        <h1 style="color: #4ECDC4;">{display_name}!</h1>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="fun-fact-box">
                        <b>💡 Gemini Fun Fact:</b> {fun_fact}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📊 See other possibilities"):
                        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
                        for idx in top_5_indices:
                            breed_name = class_names[idx]
                            # Apply override to list too
                            if breed_name in BREED_OVERRIDES:
                                breed_name = BREED_OVERRIDES[breed_name]
                            else:
                                breed_name = breed_name.replace('_', ' ').title()
                                
                            confidence = predictions[0][idx] * 100
                            st.write(f"**{breed_name}**: {confidence:.2f}%")

        except Exception as e:
            st.error(f"Oops! Something went wrong: {e}")
