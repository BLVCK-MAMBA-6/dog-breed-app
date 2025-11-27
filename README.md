# 🐶 Dog Breed Identifier

A robust and interactive dog breed classification web application powered by TensorFlow and Google Gemini. This project combines computer vision with generative AI to identify dog breeds from images and provide fun, AI-generated facts about them.

🔗 **[Live Demo](https://dog-breed-app-orhvvrbhbf9zufrkayctrq.streamlit.app/)**

## ✨ Features

- **Multi-Class Image Classification**: Identifies 120 different dog breeds using a custom-trained MobileNetV2 model.
- **Smart Pre-screening**: Uses a general object classifier to detect non-dog images and prevent incorrect classifications (e.g., cats, humans).
- **Dynamic Fun Facts**: Integrates with Google Gemini 2.5 Flash API to generate unique fun facts for each breed.
- **Confidence Thresholding**: Filters out low-confidence predictions for accuracy.
- **Breed Correction**: Handles commonly confused breeds (e.g., Husky vs. Eskimo Dog).
- **Beautiful UI**: Cute, balloon-themed, responsive interface built with Streamlit.

## 📁 Project Structure

Here is a breakdown of the key files in this repository:

- `dog_vision.ipynb`: The Jupyter Notebook used to train the machine learning model. It contains the full TensorFlow/Keras pipeline: data preprocessing, transfer learning setup (MobileNetV2), model training, and fine-tuning.
- `20251112...mobilenetv2-Adam.h5`: The saved Keras model weights. This file contains the "brain" of the AI, storing the patterns learned during training that allow it to recognize dogs.
- `app.py`: The main Python script that runs the Streamlit web application. It handles the UI, image processing, and API calls to Google Gemini.
- `class_names.json`: A JSON file containing the list of all 120 dog breeds (classes) that the model can identify, mapped to their numerical indices.
- `requirements.txt`: A list of all Python libraries (TensorFlow, Streamlit, etc.) required to run this project.

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: TensorFlow, Keras, TensorFlow Hub
- **Generative AI**: Google Gemini API (`google-generativeai`)
- **Base Model**: MobileNetV2 (Transfer Learning)
- **Language**: Python 3.10

## 🚀 How It Works

1. **Image Upload**: User uploads an image (JPG/PNG).
2. **Pre-processing**: Image gets resized and normalized for the ML model.
3. **Prediction**: Model predicts breed probability across 120 classes. If confidence < 60%, the app warns the user the image may not be a dog. If confidence ≥ 60%, the top breed prediction is shown.
4. **AI Insights**: Predicted breed name is passed to the Gemini API → returns a short fun fact.

## 📦 Installation

To run this project locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/dog-breed-app.git
    cd dog-breed-app
    ```

2. Create a virtual environment (Recommended):
    It is highly recommended to use Python 3.9 or 3.10 for TensorFlow compatibility.
    ```bash
    conda create -n dog-app python=3.10
    conda activate dog-app
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up API Keys:
    - Create a `.streamlit` folder in the root directory.
    - Create a `secrets.toml` file inside it.
    - Add your Google Gemini API key:
    ```toml
    GEMINI_API_KEY = "your_api_key_here"
    ```

5. Run the app:
    ```bash
    streamlit run app.py
    ```

## 📊 Dataset

The model was trained on the Kaggle Dog Breed Identification dataset, which is a subset of the famous Stanford Dogs Dataset.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📝 License

This project is licensed under the MIT License.
