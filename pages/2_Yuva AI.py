import streamlit as st
import json
import requests
import time

# --- Page Configuration for the AI Chat app ---
st.set_page_config(
    page_title="YuvaAI",
    layout="wide",
    initial_sidebar_state="auto",
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
        .stButton button {
            background-color: 
            color: white;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
        }
        .main-header {
            color: 
            text-align: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .st-emotion-cache-18ni341 {
            color: 
        }
        .prediction-box {
            background-color: 
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .confidence-box {
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .stSubheader {
            color: 
        }
    </style>
""", unsafe_allow_html=True)

# --- LLM Integration ---
def get_ai_explanation(query):
    apiKey = st.secrets["API_KEY"]
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
        text = candidate.get('content', {}).get('parts', [])[0].get('text', 'result')
        # Check if the returned text is just the disclaimer, indicating an issue
       # disclaimer_text = "Disclaimer: This is for informational purposes only and not a substitute for professional medical advice."
        #if text.strip() == disclaimer_text.strip():
          #  return "I was unable to generate a full explanation. The AI service may be experiencing issues or the request was not fulfilled."
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching AI explanation. Error: {e}")
    return "I am unable to provide a detailed explanation at this time. Please try again later."

# --- App UI ---
st.markdown("<h1 class='main-header' style='text-align: center;'>Yuva AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Ask the AI about the diagnosis or a general medical question.</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

# --- Display suggested questions as buttons ---
if st.session_state.suggested_questions:
    st.subheader("Suggested Questions:")
    cols = st.columns(len(st.session_state.suggested_questions))
    
    def set_prompt_from_button(question):
        st.session_state.prompt_from_button = question
    
    for i, question in enumerate(st.session_state.suggested_questions):
        with cols[i]:
            st.button(question, on_click=set_prompt_from_button, args=(question,), type="secondary")
    
    st.markdown("---")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
initial_prompt = ""
if "prompt_from_button" in st.session_state and st.session_state.prompt_from_button:
    initial_prompt = st.session_state.prompt_from_button
    del st.session_state.prompt_from_button

prompt = st.chat_input("Ask a question about brain anomalies...", key="chat_input")
if initial_prompt:
    prompt = initial_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("AI is thinking..."):
        explanation = get_ai_explanation(prompt)
        st.session_state.messages.append({"role": "assistant", "content": explanation})
    
    # Rerun to display the new message
    st.rerun()

# Display the explanation if it exists in the session state
if "explanation" in st.session_state and st.session_state.explanation:
    st.markdown(st.session_state.explanation)
