import streamlit as st

st.set_page_config(
    page_title="A Deep Learning Approach to Brain Anomalies",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="🤖"
)

# --- Custom CSS for a deep green theme with modern elements ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
            color: 
            background-color:  /* A deep, forest green background */
        }
        
        .main-header {
            color: #EAEF9D;
            text-align: center;
            font-weight: 800;
            font-size: 4em;
            letter-spacing: -2px;
            margin-bottom: 0.2em;
        }

        .subheader {
            color: #EAEF9D;
            text-align: center;
            font-weight: 400;
            font-size: 1.8em;
            margin-top: 0;
            margin-bottom: 2em;
            line-height: 1.4;
        }
        
        .st-emotion-cache-1cypcdb {
            background-color: #80B155; /* A slightly lighter green for the main container */
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
        }

        .feature-container {
            background-color: #80B155; /* Another shade of green for feature boxes */
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
            border: 2px solid transparent;
        }
        
        .feature-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.5);
            border: 2px solid #A3E4D7;
        }
        
        .feature-title {
            color: #EAEF9D;
            font-weight: 700;
            font-size: 1.5em;
            margin-top: 0;
        }
        
        .feature-description {
            color: #EAEF9D;
            font-size: 1em;
            line-height: 1.6;
        }

        .logo-text {
            color: #EAEF9D;
            font-weight: 900;
        }
        
        .st-emotion-cache-1629p25 {
            padding-top: 10px;
        }

    </style>
""", unsafe_allow_html=True)

# --- Homepage Content ---
st.markdown("<h1 class='main-header'>A Deep Learning Approach to Brain Anomalies</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subheader'>AI-powered brain health analysis and information.</h3>", unsafe_allow_html=True)

st.markdown("---")

# Use columns for a grid layout of features
col1, col2 = st.columns(2)

with col1:
    with st.container():
        #st.markdown("<div class='feature-container'>", unsafe_allow_html=True)
        st.markdown("<h4 class='feature-title'>Brain Anomaly Detector</h4>", unsafe_allow_html=True)
        # Add a "Get Started" button that redirects to the Brain Anomaly Detector page
        if st.button("Get Started", use_container_width=True):
            st.switch_page("pages/1_Brain_Anomaly_Detector.py")
        st.markdown("<p class='feature-description'>Analyze medical images to detect anomalies with high precision. Our model helps identify potential tumors, providing you with a preliminary report and detailed confidence scores.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        #st.markdown("<div class='feature-container'>", unsafe_allow_html=True)
        st.markdown("<h4 class='feature-title'>YuvaAI</h4>", unsafe_allow_html=True)
        if st.button("Get AI", use_container_width=True):
            st.switch_page("pages/2_Yuva AI.py")
        st.markdown("<p class='feature-description'>Chat with our AI health assistant for simple, non-technical explanations about your diagnosis. Get information on conditions, next steps, and professional recommendations.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

#st.markdown("<h3 style='text-align: center; color: #D1F2EB;'>Navigate using the menu on the left to get started.</h3>", unsafe_allow_html=True)
