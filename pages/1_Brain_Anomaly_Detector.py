import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import pandas as pd
import plotly.express as px
import json
import requests
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Brain Anomaly Detector",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
            color: 
            background-color:  /* A deep, forest green background */
        }
        
        /* Headers and text styling */
        .main-header {
            color: #EAEF9D;
            text-align: center;
            font-weight: 700;
            font-size: 3em;
            margin-bottom: 0.5em;
        }

        .subheader {
            color: #EAEF9D;
            text-align: center;
            font-weight: 400;
            font-size: 1.5em;
            margin-top: 0;
            margin-bottom: 2em;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #EAEF9D;
        }

        /* Buttons and interactive elements */
        .stButton button {
            background-color: 
            color: 
            font-weight: 600;
            border-radius: 10px;
            border: 2px solid 
            padding: 15px 30px;
            font-size: 1.2em;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            width: 100%;
        }

        .stButton button:hover {
            background-color: 
            color: #498428;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
            transform: translateY(-2px);
        }

        /* Main content and containers */
        .st-emotion-cache-1cypcdb {
            background-color: 
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }

        .prediction-box {
            background-color: #80B155;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .success-box {
            background-color: #1A452B;
            color: #C1D95C;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #C1D95C;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            text-align: center;
        }
        
        .warning-box {
            background-color: #1A452B;
            color: #E6B800; /* A gold color for warnings */
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #E6B800;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            text-align: center;
        }
        
        .info-box {
            background-color: rgba(193, 217, 92, 0.4);
            color: 
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
            border: 1px solid #C1D95C;
            text-align: center;
        }

        .stSubheader {
            color: #EAEF9D;
        }

        .stProgress > div > div > div > div {
            background-color: #80B155;
        }

        .upload-container {
            border: 2px dashed 
            border-radius: 15px;
            padding: 2em;
            text-align: center;
            background-color: rgba(193, 217, 92, 0.1); /* Lighter green with transparency */
        }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "page" not in st.session_state:
    st.session_state.page = "home"
if "explanation" not in st.session_state:
    st.session_state.explanation = ""

# --- Model Loading and Prediction Logic ---
@st.cache_resource
def load_model():
    """
    Loads the pre-trained Keras model from the .keras file.
    The model file must be in the same directory as this script.
    """
    try:
        model = tf.keras.models.load_model("brain_model.keras")
        return model
    except FileNotFoundError:
        st.error("Error: 'brain_model.keras' not found. Please make sure the model file is in the same folder as this script.")
        return None

def preprocess_image(image_bytes):
    """
    Preprocesses the uploaded image for model prediction.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((128, 128))
    img_array = np.expand_dims(np.array(img), axis=0) / 255.0
    return img_array

# --- LLM Integration ---

def get_ai_explanation(query):
    apiKey = st.secrets["GEMINI_API_KEY"]
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"
 # Payload for the AI API
    payload = {
        "contents": [{"parts": [{"text": query}]}],
        "tools": [{"google_search": {} }],
        "systemInstruction": {"parts": [{"text": "You are a friendly and helpful AI medical assistant. You provide simple, clear, and non-technical explanations about medical conditions. Always start your response with a clear disclaimer: 'Disclaimer: This is for informational purposes only and not a substitute for professional medical advice.'"}]}
        }
    try:
        response = requests.post(apiUrl, json=payload, timeout=60)
        response.raise_for_status() # Raise an exception for bad status codes
        result = response.json()
        candidate = result.get('candidates', [])[0]
        text = candidate.get('content', {}).get('parts', [])[0].get('text', 'No explanation found for this moment.')
        # Check if the returned text is just the disclaimer, indicating an issue
       # disclaimer_text = "Disclaimer: This is for informational purposes only and not a substitute for professional medical advice."
        #if text.strip() == disclaimer_text.strip():
          #  return "I was unable to generate a full explanation. The AI service may be experiencing issues or the request was not fulfilled."
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching AI explanation. Error: {e}")
    return "I am unable to provide a detailed explanation at this time. Please try again later."


# --- App UI Pages ---
def home_page():
    """
    Main page for the Brain Anomaly Detector.
    """
    st.markdown("<h1 class='main-header'>Brain Anomaly Detector</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Upload a medical image to check for anomalies.</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container():
            st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Choose a PNG, JPG, or JPEG file...",
                type=["png", "jpg", "jpeg"],
                help="Select a medical image of a brain to analyze."
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("")

        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
            st.success("File uploaded successfully!")

    with col2:
        if uploaded_file is not None:
            with st.spinner('Analyzing the image...'):
                model = load_model()
                if model:
                    img_array = preprocess_image(image_bytes)
                    predictions = model.predict(img_array)
                    predicted_class_index = np.argmax(predictions)
                    confidence = np.max(predictions)

                    class_labels = {
                        0: "Glioma",
                        1: "Meningioma",
                        2: "Normal",
                        3: "Pituitary"
                    }

                    predicted_label = class_labels.get(predicted_class_index, "Unknown")

                    df_predictions = pd.DataFrame({
                        'Category': list(class_labels.values()),
                        'Confidence': predictions[0]
                    })

                    if predicted_label == "Normal":
                        st.markdown("<div class='success-box'>✅ Prediction: No Tumor Detected!</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='warning-box'>⚠️ Prediction: {predicted_label} Tumor Detected!</div>", unsafe_allow_html=True)

                    st.markdown(f"<p style='color: #EAEF9D; font-weight: 600; text-align: center; margin-top: 1em;'>Confidence: `{confidence:.4f}`</p>", unsafe_allow_html=True)

                    st.markdown("---")

                    st.markdown("### Possibility of all categories", unsafe_allow_html=True)
                    fig = px.pie(
                        df_predictions,
                        values='Confidence',
                        names='Category',
                        title='Confidence by Tumor Type',
                        color_discrete_sequence=px.colors.qualitative.G10,
                        hole=.3
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color="#EAEF9D",
                        title_font_color="#EAEF9D"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    
                    st.markdown("##### 🔬 Detailed Confidence Scores", unsafe_allow_html=True)
                    det_col1, det_col2 = st.columns(2)
                    with det_col1:
                        st.markdown(f"<div class='info-box'><strong>Glioma:</strong> `{predictions[0][0]:.4f}`</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='info-box'><strong>Meningioma:</strong> `{predictions[0][1]:.4f}`</div>", unsafe_allow_html=True)
                    with det_col2:
                        st.markdown(f"<div class='info-box'><strong>Normal:</strong> `{predictions[0][2]:.4f}`</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='info-box'><strong>Pituitary:</strong> `{predictions[0][3]:.4f}`</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    st.header("Yuva AI Report")
                    st.markdown("---")
                    initial_query = f"Provide a simple explanation of a {predicted_label} tumor. What are some common medications and suggestions for a person with this condition? What kind of consultant should they seek?"
                    with st.spinner("Getting AI suggestions..."):
                        ai_explanation = get_ai_explanation(initial_query)
                    st.markdown(ai_explanation)


# The Yuva AI page and its navigation button have been removed as requested.

if st.session_state.page == "home":
    home_page()


