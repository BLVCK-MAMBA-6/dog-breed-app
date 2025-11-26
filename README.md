# 🐶 Dog Breed Identifier

A robust and interactive dog breed classification web application powered by **TensorFlow** and **Google Gemini**.  
This project combines computer vision with generative AI to identify dog breeds from images and provide fun, AI-generated facts about them.

---

## 🔗 Live Demo
*https://dog-breed-app-orhvvrbhbf9zufrkayctrq.streamlit.app/*

---

## ✨ Features

- **Multi-Class Image Classification:** Identifies **120 different dog breeds** using a custom-trained MobileNetV2 model.
- **Smart Pre-screening:** Uses a general object classifier to detect non-dog images and prevent incorrect classifications (e.g., cats, humans).
- **Dynamic Fun Facts:** Integrates with **Google Gemini 2.5 Flash API** to generate unique fun facts for each breed.
- **Confidence Thresholding:** Filters out low-confidence predictions for accuracy.
- **Breed Correction:** Handles commonly confused breeds (e.g., Husky vs. Eskimo Dog).
- **Beautiful UI:** Cute, balloon-themed, responsive interface built with **Streamlit**.

---

## 🛠 Tech Stack

- **Frontend:** Streamlit  
- **Machine Learning:** TensorFlow, Keras, TensorFlow Hub  
- **Generative AI:** Google Gemini API (`google-generativeai`)  
- **Base Model:** MobileNetV2 (Transfer Learning)  
- **Language:** Python 3.10  

---

## 🚀 How It Works

### 1. Image Upload  
User uploads an image (JPG/PNG).

### 2. Pre-processing  
Image gets resized and normalized for the ML model.

### 3. Prediction  
- Model predicts breed probability across **120 classes**.  
- If **confidence < 60%**, the app warns the user the image may not be a dog.  
- If **confidence ≥ 60%**, the top breed prediction is shown.

### 4. AI Insights  
Predicted breed name is passed to the Gemini API → returns a **short fun fact**.

---

## 📦 Installation

To run this project locally:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/dog-breed-app.git
cd dog-breed-app
