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

# --- 🧠 SMART LOGIC SETTINGS ---
# 1. The "Not a Dog" Filter
# If confidence is below 60%, we assume it's not a dog (or a very blurry one).
CONFIDENCE_THRESHOLD = 0.60  

# 2. Breed Correction Dictionary
# We map the model's technical names to friendlier, more accurate labels.
# This fixes the "Husky vs Eskimo" and "Beagle vs Foxhound" confusion.
BREED_OVERRIDES = {
    "eskimo_dog": "Husky / American Eskimo Mix",
    "siberian_husky": "Siberian Husky",
    "malamute": "Alaskan Malamute",
    "english_foxhound": "English Foxhound / Beagle", 
    "walker_hound": "Treeing Walker Coonhound",
    "blenheim_spaniel": "Cavalier King Charles Spaniel",
    "boston_bull": "Boston Terrier",
    "wire-haired_fox_terrier": "Wire Fox Terrier"
}

# --- 1. Gemini Fun Fact Generator ---
def generate_fun_fact(breed):
    """Generates a fun fact using Gemini 2.5 Flash."""
    try:
        # Retrieve key from Streamlit Secrets
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"Tell me a fun, cute, and surprising fact about the {breed} dog breed. Keep it under 40 words. Include emojis!"
            
            response = model.generate_content(prompt)
            return response.text
        else:
            return f"The {breed} is a good boy/girl! (Add API Key to secrets for dynamic facts!)"
    except Exception as e:
        return f"The {breed} is a wonderful dog! (Gemini is taking a nap right now 😴)"

# --- 2. Beautiful UI Styling ---
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
            margin-bottom: 10px;
        }
        
        .subtitle {
            text-align: center;
            color: #888;
            font-size: 1.2rem;
            margin-bottom: 30px;
        }
        
        /* Card Styling */
        .result-card {
            background-color: white;
            padding: 30px;
            border-radius: 25px;
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
            text-align: center;
            border: 4px solid #FFD93D;
            margin-top: 20px;
            margin-bottom: 20px;
            animation: fadeIn 1s;
        }
        
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        
        /* Fun Fact Box */
        .fun-fact-box {
            background-color: #E0F7FA;
            padding: 20px;
            border-radius: 15px;
            border-left: 8px solid #4ECDC4;
            margin-top: 15px;
            font-size: 1.1rem;
            color: #006064;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        
        /* Upload Widget */
        .stFileUploader { padding: 20px; }
        
        /* Image Styling */
        img { 
            border-radius: 15px; 
            border: 4px solid #FF6B6B; 
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        
        /* Button Styling */
        .stButton button {
            background-color: #FF6B6B;
            color: white;
            border-radius: 12px;
            font-family: 'Fredoka One', cursive;
            border: none;
            padding: 10px 24px;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #FF5252;
            transform: scale(1.05);
            transition: all 0.2s;
        }
        </style>
    """, unsafe_allow_html=True)

add_custom_css()

# --- 3. Model Architecture ---
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

# --- 4. Load Resources ---
@st.cache_resource
def load_model_and_classes():
    try:
        model = create_model(MODULE_HANDLE, NUM_CLASSES)
        model.load_weights(WEIGHTS_FILE, by_name=True)
        with open(CLASSES_FILE, 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Critical Error: {e}")
        return None, None

model, class_names = load_model_and_classes()

# --- 5. Preprocessing ---
def preprocess_image(image_pil):
    image = image_pil.resize((IMG_SIZE, IMG_SIZE)) 
    image_array = np.array(image)
    if len(image_array.shape) == 2: image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4: image_array = image_array[:, :, :3]
    image_array = image_array.astype(np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

# --- 6. Main App Logic ---
st.title("🐾 Dog Breed Identifier 🐾")
st.markdown('<div class="subtitle">Upload a photo and let AI guess the breed! 📸</div>', unsafe_allow_html=True)

if model and class_names:
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            # Layout: Image on top/center
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with st.spinner('🧠 AI is thinking...'):
                preprocessed_image = preprocess_image(image)
                predictions = model.predict(preprocessed_image, verbose=0)
                
                top_prediction_index = np.argmax(predictions)
                top_confidence = predictions[0][top_prediction_index]
                raw_breed = class_names[top_prediction_index]
                
                # --- SMART CHECK: Is it a dog? ---
                if top_confidence < CONFIDENCE_THRESHOLD:
                    st.warning(f"⚠️ **I'm unsure ({top_confidence*100:.1f}% confidence).**")
                    st.info("This might not be a dog, or it's a mixed breed I haven't studied yet. Try a clearer photo!")
                
                else:
                    # --- SUCCESS! ---
                    
                    # 1. Apply Smart Overrides (Fixing confusions)
                    if raw_breed in BREED_OVERRIDES:
                        display_name = BREED_OVERRIDES[raw_breed]
                    else:
                        display_name = raw_breed.replace('_', ' ').title()

                    # 2. Get Fun Fact from Gemini
                    fun_fact = generate_fun_fact(display_name)

                    # 3. Celebration
                    st.balloons()
                    
                    # 4. Result Card
                    st.markdown(f"""
                    <div class="result-card">
                        <h2 style="margin:0;">It looks like a...</h2>
                        <h1 style="color: #4ECDC4; font-size: 2.5rem;">{display_name}</h1>
                        <p style="color: #666;">Confidence: <b>{top_confidence*100:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 5. Fun Fact Card
                    st.markdown(f"""
                    <div class="fun-fact-box">
                        <b>🤖 Gemini says:</b> {fun_fact}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 6. Expandable Stats
                    with st.expander("📊 View Detailed Analysis"):
                        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
                        for idx in top_5_indices:
                            breed = class_names[idx]
                            # Show the friendly name in stats too
                            name = BREED_OVERRIDES.get(breed, breed.replace('_', ' ').title())
                            conf = predictions[0][idx] * 100
                            st.progress(int(conf))
                            st.write(f"**{name}**: {conf:.2f}%")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
