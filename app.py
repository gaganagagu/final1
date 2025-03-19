import streamlit as st
from main import ann_app

@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('model.h5')

import tensorflow as tf
loaded_model = tf.keras.models.load_model('model.h5')

from auth import register_user, authenticate_user  # Import authentication functions

# Initialize session state for login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""


def main():
    # Custom CSS for better styling
    
    st.markdown("""
        <style>
        .main-title {
            font-size: 34px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(to right, #1E88E5, #1565C0);
            color: white;
            border-radius: 10px;
        }
        .section-header {
            font-size: 24px;
            font-weight: bold;
            color: #1E88E5;
            margin: 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #1E88E5;
        }
        .sidebar-header {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }
        .about-box {
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #1E88E5;
            margin: 10px 0;
        }
        .future-scope-item {
            padding: 15px;
            background-color: #f0f8ff;
            border-radius: 8px;
            margin: 10px 0;
        }
        .creator-info {
            text-align: center;
            padding: 20px;
            background-color: #e3f2fd;
            border-radius: 10px;
            margin-top: 30px;
        }
        .highlight-text {
            color: #1E88E5;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main title with enhanced styling
    st.markdown('<p class="main-title">HEART WISE: AI-DRIVEN HEART DISEASE PREDICTION USING NEURAL NETWORKS</p>', 
                unsafe_allow_html=True)

    # Enhanced sidebar
    with st.sidebar:
        st.markdown('<p class="sidebar-header">Navigation Menu</p>', unsafe_allow_html=True)
        menu = ["Login", "Register", "Home", "Model", "Metrics", "About"]

        # Remove login/register if user is already logged in
        if st.session_state.logged_in:
            menu.remove("Login")
            menu.remove("Register")

        choice = st.selectbox('', menu)

    if choice == "Login":
        st.title("🔐 User Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"✅ Welcome, {username}!")
                st.rerun()  # Refresh page after login
            else:
                st.error("❌ Invalid username or password.")

    elif choice == "Register":
        st.title("🔑 User Registration")

        new_username = st.text_input("Choose a Username")
        new_password = st.text_input("Choose a Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match! 🔴")
            elif register_user(new_username, new_password):
                st.success("✅ Registration successful! Please go to the Login page.")
            else:
                st.error("❌ Username already exists. Try a different one.")


    if choice == 'Home':
        if not st.session_state.logged_in:
            st.warning("⚠️ Please login to access the Home page.")
            st.stop()  # Blocks unauthorized users

        st.markdown('<p class="section-header">Introduction</p>', unsafe_allow_html=True)
        st.write(f"👤 Logged in as: **{st.session_state.username}**")

        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.experimental_rerun()

        
        st.markdown('<p class="section-header">Introduction</p>', unsafe_allow_html=True)
        
        # Create tabs for different sections of information
        tabs = st.tabs(["Overview", "Dataset Features"])
        
        with tabs[0]:
            st.markdown("""
            Heart disease prediction using Artificial Neural Networks (ANN) is a cutting-edge approach in medical diagnosis 
            and research. This application demonstrates how machine learning can analyze patient health data to predict 
            the likelihood of heart disease.
            """)
            
            # Add an info box
            st.info("📊 This model analyzes multiple health parameters to provide risk assessment for heart disease.")
            
            # First part of the original text
            st.write("""
            ANN is a sophisticated machine learning technique that can identify complex patterns and relationships within medical data 
            to make accurate predictions or classifications. In the context of heart disease prediction, our ANN model analyzes 
            various patient health indicators to assess the risk of developing heart disease.
            """)

        with tabs[1]:
            # Create an organized feature list with better formatting
            features = {
                "Patient Demographics": {
                    "AGE": "Patient's age - a crucial factor in heart disease risk",
                    "GENDER": "Patient's gender - impacts risk factors and disease manifestation"
                },
                "Vital Signs": {
                    "RESTING_BP": "Resting blood pressure - key indicator of cardiovascular health",
                    "MAX_HEART_RATE": "Maximum heart rate achieved - important stress indicator"
                },
                "Blood Tests": {
                    "SERUM_CHOLESTEROL": "Total cholesterol levels in blood",
                    "TRI_GLYCERIDE": "Triglyceride levels - fat type linked to heart disease",
                    "LDL": "Low-density lipoprotein - 'bad' cholesterol",
                    "HDL": "High-density lipoprotein - 'good' cholesterol",
                    "FBS": "Fasting blood sugar - diabetes indicator"
                },
                "Diagnostic Tests": {
                    "CHEST_PAIN": "Type and severity of chest pain",
                    "RESTING_ECG": "Resting electrocardiogram results",
                    "ECHO": "Echocardiogram findings",
                    "TMT": "Treadmill Test results"
                }
            }

            for category, items in features.items():
                st.markdown(f"**{category}**")
                for feature, description in items.items():
                    st.markdown(f"- **{feature}**: {description}")
                st.write("")

    elif choice == 'Model':
        if not st.session_state.logged_in:
            st.warning("⚠️ Please login to access the Model page.")
            st.stop()
        ann_app()
        
    elif choice == "Metrics":
        if not st.session_state.logged_in:
            st.warning("⚠️ Please login to access the Metrics page.")
            st.stop()
        st.markdown('<p class="section-header">Model Metrics and Performance</p>', unsafe_allow_html=True)
        with open('exp.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800)
        f.close()
   
    else:  # About section
        if not st.session_state.logged_in:
            st.warning("⚠️ Please login to access the About page.")
            st.stop()
        st.markdown('<p class="section-header">About the Project</p>', unsafe_allow_html=True)
        
        # Project significance
        with st.expander("🎯 Project Significance", expanded=True):
            st.markdown("""
            <div class="about-box">
            This project represents a significant contribution to healthcare technology by:
            - 🏥 Addressing a leading global health concern
            - 🔍 Enabling early detection of heart disease
            - 💻 Leveraging advanced machine learning techniques
            - 🤝 Bridging healthcare and technology
            - 📈 Providing data-driven insights for medical professionals
            </div>
            """, unsafe_allow_html=True)

        # Future scope with better organization
        st.markdown('<p class="section-header">Future Scope</p>', unsafe_allow_html=True)
        
        future_scope_items = [
            ("Model Enhancement", "Refining the predictive model with advanced ANN architectures and optimization techniques"),
            ("Healthcare Integration", "Seamless integration with existing healthcare systems and EHR"),
            ("Mobile Solutions", "Development of mobile applications and wearable device integration"),
            ("Advanced Analytics", "Incorporation of genetic data and advanced imaging analysis"),
            ("Clinical Support", "Enhanced decision support tools for healthcare professionals"),
            ("Outcome Prediction", "Long-term outcome predictions and risk assessment"),
            ("Data Integration", "Integration with multiple data sources for comprehensive analysis")
        ]

        cols = st.columns(2)
        for i, (title, description) in enumerate(future_scope_items):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="future-scope-item">
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """, unsafe_allow_html=True)

main()
