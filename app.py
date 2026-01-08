# Libraries Import

import numpy as np
import pandas as pd
import re
import pickle
import os

import streamlit as st

# MUST be first Streamlit command
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
    "The app analyzes overall sentiment and generates insights."
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
    df["sentiment_code"] = model.predict(X_tfidf)
    df["sentiment_label"] = df["sentiment_code"].map(SENTIMENT_MAP)

    # Prediction Confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_tfidf)
        df["confidence"] = proba.max(axis=1)
    else:
        df["confidence"] = np.nan

# Overall Sentiment Insight

    st.subheader("Overall Sentiment Insight")

    sentiment_percent = (
        df["sentiment_label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    st.dataframe(
        sentiment_percent.reset_index().rename(
            columns={"index": "Sentiment", "sentiment_label": "Percentage"}
        )
    )

    dominant_sentiment = sentiment_percent.idxmax()

    if dominant_sentiment == "Positive":
        st.success(f"Overall Twitter Sentiment: {dominant_sentiment}")
    elif dominant_sentiment == "Negative":
        st.error(f"Overall Twitter Sentiment: {dominant_sentiment}")
    else:
        st.warning(f"Overall Twitter Sentiment: {dominant_sentiment}")

# TOP POSITIVE & NEGATIVE TWEETS (BUTTON)

    st.subheader("Top Tweets by Sentiment")

    if st.button("Generate Top 5 Positive & Negative Tweets"):

        col_pos, col_neg = st.columns(2)

        # ---------- Top Positive ----------
        with col_pos:
            st.markdown("### 😊 Top Positive Tweets")

            top_positive = (
                df[df["sentiment_label"] == "Positive"]
                .sort_values(by="confidence", ascending=False)
                .head(5)
            )

            if top_positive.empty:
                st.info("No positive tweets found.")
            else:
                for _, row in top_positive.iterrows():
                    st.markdown(
                        f"- **Confidence:** {round(row['confidence']*100, 2)}%  \n"
                        f"- {row[text_column]}"
                    )

        # ---------- Top Negative ----------
        with col_neg:
            st.markdown("### 😠 Top Negative Tweets")

            top_negative = (
                df[df["sentiment_label"] == "Negative"]
                .sort_values(by="confidence", ascending=False)
                .head(5)
            )

            if top_negative.empty:
                st.info("No negative tweets found.")
            else:
                for _, row in top_negative.iterrows():
                    st.markdown(
                        f"- **Confidence:** {round(row['confidence']*100, 2)}%  \n"
                        f"{row[text_column]}"
                    )

# WORD CLOUD SECTION

    st.subheader("Sentiment-wise Word Clouds")

    if st.button("Generate Word Clouds"):

        col1, col2, col3 = st.columns(3)

        def generate_wordcloud(text, title, column):
            if text.strip() == "":
                column.info(f"No words available for {title}")
                return

            wc = WordCloud(
                width=400,
                height=300,
                background_color="white"
            ).generate(text)

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            column.pyplot(fig)
            column.caption(title)

        positive_text = " ".join(
            df[df["sentiment_label"] == "Positive"]["processed_text"]
        )
        neutral_text = " ".join(
            df[df["sentiment_label"] == "Neutral"]["processed_text"]
        )
        negative_text = " ".join(
            df[df["sentiment_label"] == "Negative"]["processed_text"]
        )

        generate_wordcloud(positive_text, "Positive Tweets", col1)
        generate_wordcloud(neutral_text, "Neutral Tweets", col2)
        generate_wordcloud(negative_text, "Negative Tweets", col3)


# =========================================
# Footer
# =========================================
st.markdown("---")
st.caption(
    "Disclaimer: This application is built strictly for academic purposes. "
    "Sentiment predictions are generated using a machine learning model and may not be 100% accurate."
)
