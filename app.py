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

# --- 2. Enhanced Custom CSS ---
def add_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Space+Grotesk:wght@500;700&display=swap');
        
        /* Dark minimalist background */
        .stApp {
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
            font-family: 'Inter', sans-serif;
        }
        
        /* Clean title styling */
        h1 { 
            font-family: 'Space Grotesk', sans-serif; 
            color: #e0e0e0;
            text-align: center; 
            font-size: 3rem;
            margin-bottom: 0;
            font-weight: 700;
            letter-spacing: -1px;
        }
        
        h2 { 
            color: #60a5fa; 
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
        }
        
        h4 {
            color: #9ca3af;
        }
        
        /* Sleek dark card */
        .result-card {
            background: rgba(30, 30, 46, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(96, 165, 250, 0.3);
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            text-align: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        .result-card h2 {
            font-size: 1.1rem;
            margin-bottom: 8px;
        }
        
        .result-card h1 {
            font-size: 2.2rem;
            margin-top: 0;
        }
        
        /* Modern info box */
        .fun-fact-box {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(96, 165, 250, 0.3);
            padding: 16px;
            border-radius: 12px;
            border-left: 4px solid #60a5fa;
            margin-top: 16px;
            font-size: 0.95rem;
            color: #d1d5db;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        }
        
        .fact-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .refresh-btn {
            background: rgba(96, 165, 250, 0.2);
            border: 1px solid rgba(96, 165, 250, 0.4);
            color: #60a5fa;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .refresh-btn:hover {
            background: rgba(96, 165, 250, 0.3);
            border-color: rgba(96, 165, 250, 0.6);
        }
        
        /* Clean file uploader */
        .stFileUploader { 
            text-align: center;
        }
        
        .stFileUploader > div {
            background: rgba(30, 30, 46, 0.6);
            border-radius: 12px;
            padding: 20px;
            border: 2px solid rgba(96, 165, 250, 0.3);
            transition: border-color 0.2s ease;
        }
        
        .stFileUploader > div:hover {
            border-color: rgba(96, 165, 250, 0.6);
        }
        
        /* Refined image display */
        img { 
            border-radius: 12px; 
            border: 1px solid rgba(96, 165, 250, 0.3);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }
        
        /* Sophisticated button */
        .stButton button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            padding: 10px 24px;
            border: none;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
        
        /* Minimalist expander */
        .streamlit-expanderHeader {
            background: rgba(30, 30, 46, 0.6);
            border-radius: 8px;
            padding: 12px;
            font-weight: 600;
            color: #9ca3af;
            border: 1px solid rgba(96, 165, 250, 0.2);
            transition: border-color 0.2s ease;
        }
        
        .streamlit-expanderHeader:hover {
            border-color: rgba(96, 165, 250, 0.5);
        }
        
        /* Spinner */
        .stSpinner > div {
            border-color: #60a5fa transparent transparent transparent !important;
        }
        
        /* Alert boxes */
        .stAlert {
            border-radius: 8px;
            border: 1px solid rgba(96, 165, 250, 0.3);
            background: rgba(30, 30, 46, 0.8);
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
st.title("🐾 Dog Breed Identifier")
st.markdown("<h4 style='text-align: center; color: #888;'>Upload a photo and I'll identify the breed 📸</h4>", unsafe_allow_html=True)

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
                    st.warning(f"🤔 Only **{top_confidence*100:.1f}%** confident about this prediction.")
                    st.info("This might not be a dog, or it's an uncommon breed. Try a clearer photo.")
                else:
                    # Apply Manual Overrides (Fixing Husky/Eskimo confusion)
                    if raw_breed in BREED_OVERRIDES:
                        display_name = BREED_OVERRIDES[raw_breed]
                    else:
                        display_name = raw_breed.replace('_', ' ').title()

                    # Store display name in session state for regeneration
                    if 'current_breed' not in st.session_state or st.session_state.current_breed != display_name:
                        st.session_state.current_breed = display_name
                        st.session_state.fun_fact = generate_fun_fact(display_name)
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h2>{top_confidence*100:.0f}% confident it's a</h2>
                        <h1 style="color: #60a5fa;">{display_name}</h1>
                    </div>
                    """, unsafe_allow_html=True)

                    # Fun fact with refresh button
                    col_fact, col_btn = st.columns([4, 1])
                    with col_fact:
                        st.markdown(f"""
                        <div class="fun-fact-box">
                            <b>💡 Fun Fact:</b> {st.session_state.fun_fact}
                        </div>
                        """, unsafe_allow_html=True)
                    with col_btn:
                        if st.button("🔄", help="Get new fact"):
                            st.session_state.fun_fact = generate_fun_fact(display_name)
                            st.rerun()
                    
                    with st.expander("📊 Other possibilities"):
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
