import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="News Authenticity Intelligence System",
    page_icon="📰",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- CLEAN FUNCTION ---------------- #
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("📊 System Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["News Analyzer", "Model Insights", "About Project"]
)

# ---------------- PAGE 1: ANALYZER ---------------- #
if page == "News Analyzer":

    st.title("📰 News Authenticity Intelligence System")
    st.markdown("### AI-Powered Fake News Detection Platform")

    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area(
            "Enter News Article Text",
            height=250
        )

        if st.button("Analyze News"):
            if user_input.strip() == "":
                st.warning("Please enter some news content.")
            else:
                cleaned = clean_text(user_input)
                vectorized = vectorizer.transform([cleaned])

                prediction = model.predict(vectorized)
                probability = model.predict_proba(vectorized)

                confidence = round(np.max(probability) * 100, 2)

                st.markdown("---")

                if prediction[0] == 1:
                    st.success("Prediction: REAL News")
                else:
                    st.error("Prediction: FAKE News")

                st.metric("Confidence Score", f"{confidence}%")
                st.progress(int(confidence))

                # Explainability
                feature_names = np.array(vectorizer.get_feature_names_out())
                sorted_indices = np.argsort(vectorized.toarray()).flatten()[::-1]
                top_words = feature_names[sorted_indices][:5]

                st.markdown("### Top Contributing Keywords")
                st.write(", ".join(top_words))

    with col2:
        st.markdown("### Model Information")
        st.info("""
        • Algorithm: Logistic Regression  
        • Feature Extraction: TF-IDF  
        • Problem Type: Binary Classification  
        • Classes: Real (1) / Fake (0)
        """)

# ---------------- PAGE 2: MODEL INSIGHTS ---------------- #
elif page == "Model Insights":

    st.title("📈 Model Performance Insights")

    st.markdown("""
    ### Evaluation Metrics

    - Accuracy: ~98%
    - Precision: High reliability for fake detection
    - Recall: Strong detection coverage
    - F1-Score: Balanced performance
    """)

    st.markdown("""
    ### Why Logistic Regression?
    Logistic Regression performs efficiently on high-dimensional sparse data like TF-IDF vectors.
    """)

# ---------------- PAGE 3: ABOUT ---------------- #
else:
    st.title("🏢 About This Project")

    st.markdown("""
    ### Project Overview
    This system analyzes textual news content and classifies it as REAL or FAKE using supervised machine learning.

    ### Architecture
    1. Text Cleaning
    2. TF-IDF Vectorization
    3. Logistic Regression Classification
    4. Probability-Based Confidence Scoring

    ### Deployment
    Built using:
    - Python
    - Scikit-learn
    - Streamlit

    ### Use Cases
    - Media verification
    - Journalism support
    - Academic research
    - Social media content moderation
    """)

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("© 2026 News Authenticity Intelligence System | Built with Machine Learning")