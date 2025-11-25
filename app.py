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
BREED_OVERRIDES = {
    "eskimo_dog": "American Eskimo / Husky Mix",
    "blenheim_spaniel": "Cavalier King Charles Spaniel",
    "boston_bull": "Boston Terrier"
}

# --- 1. Gemini Fun Fact Generator ---
def generate_fun_fact(breed):
    """Generates a fun fact using the app's shared Gemini API key."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"Give me one short, fun, and cute fact about the {breed} dog breed. Keep it under 30 words. Use emojis!"
            
            response = model.generate_content(prompt)
            return response.text
        else:
            return f"The {breed} is a good boy/girl! (API Key missing for fresh facts!)"
    except Exception as e:
        return f"The {breed} is a good boy/girl! (We couldn't fetch a new fact right now, but they are amazing!)"

# --- 2. Enhanced Custom CSS (THE FIX) ---
def add_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        /* 1. BACKGROUND: Soft Off-Black instead of Pitch Black */
        .stApp {
            background-color: #121212; 
            font-family: 'Inter', sans-serif;
            color: #E0E0E0;
        }
        
        /* 2. HEADER STYLING */
        h1 {
            font-weight: 600;
            letter-spacing: -0.05em;
            color: #FFFFFF;
            font-size: 2.5rem !important;
            margin-bottom: 0.5rem;
        }
        
        /* 3. COMPACT RESULT CARD (The "Too Large" Fix) */
        .result-card {
            background-color: #1E1E1E;
            border: 1px solid #333333;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            text-align: left; /* Aligned left looks more mature */
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            border-left: 4px solid #6366F1; /* Indigo accent */
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .result-info {
            display: flex;
            flex-direction: column;
        }

        .confidence-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #9CA3AF;
            margin-bottom: 4px;
        }

        .breed-title {
            font-size: 1.5rem; /* Smaller, cleaner size */
            font-weight: 600;
            color: #FFFFFF;
            margin: 0;
        }

        .confidence-badge {
            background-color: #312E81;
            color: #A5B4FC;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        /* 4. FUN FACT BOX */
        .fun-fact-box {
            background-color: #18181b;
            border: 1px solid #27272a;
            padding: 1rem;
            border-radius: 8px;
            color: #d4d4d8;
            font-size: 0.95rem;
            line-height: 1.5;
            margin-top: 1rem;
        }

        /* 5. UPLOAD AREA CLEANUP */
        .stFileUploader > div > div {
            background-color: #1E1E1E;
            border: 1px dashed #4B5563;
        }
        
        .stFileUploader > div > div:hover {
            border-color: #6366F1;
        }

        /* 6. BUTTONS */
        .stButton button {
            background-color: #6366F1;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-weight: 500;
            transition: background 0.3s;
        }
        .stButton button:hover {
            background-color: #4F46E5;
        }

        /* 7. IMAGE BORDER */
        img {
            border-radius: 8px;
            border: 1px solid #333;
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
    if len(image_array.shape) == 2: image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4: image_array = image_array[:, :, :3]
    
    image_array = image_array.astype(np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

# --- 6. Main App ---
st.title("Dog Breed Identifier")
st.markdown("<p style='text-align: center; color: #9CA3AF; margin-top: -10px;'>AI-powered canine classification</p>", unsafe_allow_html=True)

if model is None or class_names is None:
    st.error("⚠️ Could not load model. Please check files.")
else:
    # Use columns to center the uploader visually
    col_spacer_l, col_main, col_spacer_r = st.columns([1, 4, 1])
    
    with col_main:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            # Layout: Image on Left, Results on Right (Desktop) or Stacked (Mobile)
            # This looks much more mature than stacking everything vertically
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                with st.spinner('Analyzing patterns...'):
                    preprocessed_image = preprocess_image(image)
                    predictions = model.predict(preprocessed_image, verbose=0)
                    
                    top_prediction_index = np.argmax(predictions)
                    top_confidence = predictions[0][top_prediction_index]
                    raw_breed = class_names[top_prediction_index]
                    
                    # Logic: Is it a dog?
                    if top_confidence < CONFIDENCE_THRESHOLD:
                        st.warning(f"Uncertain prediction ({top_confidence*100:.1f}%).")
                        st.caption("Try a clearer photo or a different angle.")
                    else:
                        # Breed Name Logic
                        if raw_breed in BREED_OVERRIDES:
                            display_name = BREED_OVERRIDES[raw_breed]
                        else:
                            display_name = raw_breed.replace('_', ' ').title()

                        # Session State for Facts
                        if 'current_breed' not in st.session_state or st.session_state.current_breed != display_name:
                            st.session_state.current_breed = display_name
                            st.session_state.fun_fact = generate_fun_fact(display_name)
                        
                        # --- NEW COMPACT RESULT CARD ---
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-info">
                                <span class="confidence-label">Best Match</span>
                                <h3 class="breed-title">{display_name}</h3>
                            </div>
                            <div class="confidence-badge">
                                {top_confidence*100:.0f}% Match
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Fun Fact
                        st.markdown(f"""
                        <div class="fun-fact-box">
                            <b>💡 AI Insight:</b> {st.session_state.fun_fact}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("Regenerate Fact", key="refresh_fact"):
                            st.session_state.fun_fact = generate_fun_fact(display_name)
                            st.rerun()
                            
                        # Other possibilities expander
                        with st.expander("See Alternative Matches"):
                            top_5_indices = np.argsort(predictions[0])[-5:][::-1]
                            for idx in top_5_indices:
                                if idx == top_prediction_index: continue # Skip the top one
                                b_name = class_names[idx]
                                b_name = BREED_OVERRIDES.get(b_name, b_name.replace('_', ' ').title())
                                conf = predictions[0][idx] * 100
                                st.progress(int(conf), text=f"{b_name} ({conf:.1f}%)")

        except Exception as e:
            st.error(f"Error processing image: {e}")
