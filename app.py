# Imports
import numpy as np
import pandas as pd
import re
import pickle
import os

import streamlit as st

st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    layout="wide"
)

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Download NLTK Resources

@st.cache_resource
def download_nltk_resources():
    nltk.download("stopwords")
    nltk.download("wordnet")

download_nltk_resources()

# Load Model & Vectorizer

@st.cache_resource
def load_artifacts():
    if not os.path.exists("model/tfidf.pkl") or not os.path.exists("model/sentiment_model.pkl"):
        st.error("Model files not found. Please ensure model/tfidf.pkl and model/sentiment_model.pkl exist.")
        st.stop()

    with open("model/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("model/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)

    return tfidf, model

tfidf, model = load_artifacts()

# Sentiment Label Mapping

SENTIMENT_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
# Text Preprocessing

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



# Streamlit UI

st.title("Twitter Sentiment Analysis using NLP")
st.write(
    "Upload a CSV or Excel file containing Twitter reviews/comments. "
    "The app will analyze the overall sentiment and generate insights."
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

# Read File 
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

 # Select Text Column
    text_column = st.selectbox(
        "Select the column containing Twitter text",
        df.columns
    )

 # Preprocessing
    df["clean_text"] = df[text_column].apply(clean_text)
    df["processed_text"] = df["clean_text"].apply(preprocess_text)

# Vectorization
    X_tfidf = tfidf.transform(df["processed_text"])

# Prediction
    df["predicted_sentiment"] = model.predict(X_tfidf)
    df["predicted_sentiment_label"] = df["predicted_sentiment"].map(SENTIMENT_MAP)

# Overall Sentiment Insight
    st.subheader("Overall Sentiment Insight")

    sentiment_counts = df["predicted_sentiment_label"].value_counts()
    sentiment_percent = (sentiment_counts / sentiment_counts.sum() * 100).round(2)

    st.dataframe(sentiment_percent.reset_index().rename(
        columns={"index": "Sentiment", "predicted_sentiment_label": "Percentage"}
    ))

    dominant_sentiment = sentiment_percent.idxmax()

    if dominant_sentiment.lower() == "positive":
        st.success(f"Overall Twitter Sentiment: {dominant_sentiment}")
    elif dominant_sentiment.lower() == "negative":
        st.error(f"Overall Twitter Sentiment: {dominant_sentiment}")
    else:
        st.warning(f"Overall Twitter Sentiment: {dominant_sentiment}")

# Generating Word Cloud
    st.subheader("Word Cloud")

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

# =========================================
# Footer
# =========================================
st.markdown("---")
st.caption(
    "Disclaimer: This application is built strictly for academic purposes. "
    "Sentiment predictions are generated using a machine learning model and may not be 100% accurate."
)
