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
        @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;700;800&display=swap');
        
        /* Animated gradient background */
        .stApp {
            background: linear-gradient(-45deg, #FFF5F7, #FFE4E1, #E8F5E9, #FFF9C4);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            font-family: 'Nunito', sans-serif;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Floating paw prints */
        .stApp::before {
            content: "🐾";
            position: fixed;
            font-size: 3rem;
            opacity: 0.1;
            animation: float 6s ease-in-out infinite;
            top: 10%;
            left: 10%;
        }
        
        .stApp::after {
            content: "🐾";
            position: fixed;
            font-size: 2.5rem;
            opacity: 0.1;
            animation: float 8s ease-in-out infinite;
            top: 70%;
            right: 15%;
            animation-delay: 2s;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(10deg); }
        }
        
        /* Title with enhanced styling */
        h1 { 
            font-family: 'Fredoka One', cursive; 
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #FFD93D 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center; 
            text-shadow: 3px 3px 0px rgba(255, 230, 109, 0.3); 
            font-size: 3.5rem;
            margin-bottom: 0;
            animation: titlePulse 2s ease-in-out infinite;
        }
        
        @keyframes titlePulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        
        h2 { 
            color: #4ECDC4; 
            font-family: 'Fredoka One', cursive;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        h4 {
            animation: fadeIn 1s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Enhanced card with gradient border */
        .result-card {
            background: linear-gradient(white, white) padding-box,
                        linear-gradient(135deg, #FF6B6B, #4ECDC4, #FFD93D) border-box;
            border: 5px solid transparent;
            padding: 30px;
            border-radius: 25px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.15), 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 25px;
            margin-bottom: 25px;
            animation: cardSlideUp 0.5s ease-out;
            position: relative;
            overflow: hidden;
        }
        
        .result-card::before {
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.3), transparent);
            transform: rotate(45deg);
            animation: shine 3s infinite;
        }
        
        @keyframes shine {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
        
        @keyframes cardSlideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Enhanced fun fact box */
        .fun-fact-box {
            background: linear-gradient(135deg, #E0F7FA 0%, #B2EBF2 100%);
            padding: 25px;
            border-radius: 20px;
            border-left: 10px solid #4ECDC4;
            margin-top: 20px;
            font-style: italic;
            font-size: 1.15rem;
            color: #006064;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            animation: factFadeIn 0.6s ease-out 0.3s both;
            position: relative;
        }
        
        .fun-fact-box::before {
            content: "✨";
            position: absolute;
            top: 10px;
            right: 15px;
            font-size: 2rem;
            animation: sparkle 1.5s ease-in-out infinite;
        }
        
        @keyframes sparkle {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.2); }
        }
        
        @keyframes factFadeIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        /* Enhanced file uploader */
        .stFileUploader { 
            text-align: center;
        }
        
        .stFileUploader > div {
            background: white;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.08);
            border: 3px dashed #FF6B6B;
            transition: all 0.3s ease;
        }
        
        .stFileUploader > div:hover {
            border-color: #4ECDC4;
            box-shadow: 0 12px 28px rgba(0,0,0,0.12);
            transform: translateY(-2px);
        }
        
        /* Enhanced image display */
        img { 
            border-radius: 20px; 
            border: 5px solid transparent;
            background: linear-gradient(white, white) padding-box,
                       linear-gradient(135deg, #FF6B6B, #FFD93D) border-box;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            transition: transform 0.3s ease;
        }
        
        img:hover {
            transform: scale(1.02);
        }
        
        /* Enhanced button styling */
        .stButton button {
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
            color: white;
            border-radius: 15px;
            font-family: 'Fredoka One', cursive;
            padding: 12px 30px;
            border: none;
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.3);
            transition: all 0.3s ease;
            font-size: 1.1rem;
        }
        
        .stButton button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(255, 107, 107, 0.4);
        }
        
        /* Enhanced expander */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
            border-radius: 15px;
            padding: 15px;
            font-weight: 700;
            color: #495057;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, #E9ECEF 0%, #DEE2E6 100%);
            transform: translateX(5px);
        }
        
        /* Spinner enhancement */
        .stSpinner > div {
            border-color: #FF6B6B transparent transparent transparent !important;
        }
        
        /* Warning and info boxes */
        .stAlert {
            border-radius: 15px;
            border: none;
            box-shadow: 0 6px 16px rgba(0,0,0,0.1);
            animation: alertSlideIn 0.4s ease-out;
        }
        
        @keyframes alertSlideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
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
