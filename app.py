import numpy as np
import pandas as pd
import re
import pickle

import streamlit as st

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# WordCloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# =========================================
# Load Model & Vectorizer
# =========================================

@st.cache_resource
def load_artifacts():
    with open("model/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("model/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

tfidf, model = load_artifacts()

# =========================================
# Text Preprocessing Functions
# =========================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# =========================================
# Streamlit UI
# =========================================

st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide")

st.title("🐦 Twitter Sentiment Analysis using NLP")
st.write(
    "Upload a CSV or Excel file containing Twitter reviews/comments to analyze overall sentiment."
)

# =========================================
# File Upload
# =========================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # -------------------------------------
    # Read File
    # -------------------------------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # -------------------------------------
    # Column Selection
    # -------------------------------------
    text_column = st.selectbox(
        "Select the column containing Twitter text",
        df.columns
    )

    # -------------------------------------
    # Text Processing
    # -------------------------------------
    df["clean_text"] = df[text_column].apply(clean_text)
    df["processed_text"] = df["clean_text"].apply(preprocess_text)

    # -------------------------------------
    # TF-IDF Transformation
    # -------------------------------------
    X_tfidf = tfidf.transform(df["processed_text"])

    # -------------------------------------
    # Prediction
    # -------------------------------------
    df["predicted_sentiment"] = model.predict(X_tfidf)

    # -------------------------------------
    # Overall Sentiment Insight
    # -------------------------------------
    st.subheader("📊 Overall Sentiment Distribution")

    sentiment_counts = df["predicted_sentiment"].value_counts()
    sentiment_percent = round(sentiment_counts / sentiment_counts.sum() * 100, 2)

    st.write(sentiment_percent)

    dominant_sentiment = sentiment_percent.idxmax()

    st.success(f"**Overall Twitter Sentiment:** {dominant_sentiment}")

    # -------------------------------------
    # Word Cloud
    # -------------------------------------
    st.subheader("☁️ Word Cloud from Tweets")

    combined_text = " ".join(df["processed_text"])

    wordcloud = WordCloud(
        width=900,
        height=400,
        background_color="white"
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)

    # -------------------------------------
    # Download Results
    # -------------------------------------
    st.subheader("⬇️ Download Predictions")

    output_df = df[[text_column, "predicted_sentiment"]]
    csv = output_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Sentiment Results",
        data=csv,
        file_name="twitter_sentiment_output.csv",
        mime="text/csv"
    )

# =========================================
# Footer
# =========================================

st.markdown("---")
st.caption(
    "⚠️ This application is built for academic purposes. "
    "Sentiment predictions are based on a trained NLP model and may not be 100% accurate."
)
