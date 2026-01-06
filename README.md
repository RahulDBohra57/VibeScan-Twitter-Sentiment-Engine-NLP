App Link: https://nlp-twitter-sentiment-analysis.streamlit.app/

# NLP-Pipeline-Twitter-Sentiment-Analysis

📌 Project Overview

This project is a Natural Language Processing (NLP) based Twitter Sentiment Analysis application that analyzes user tweets or comments and determines whether the overall sentiment is Positive, Neutral, or Negative.

The system uses text preprocessing, TF-IDF feature extraction, and machine learning classification to predict sentiment. A Streamlit web application is built on top of the trained model to allow users to upload their own Twitter data and view insights interactively.


🚀 Streamlit Application Workflow
- User uploads a CSV or Excel file containing Twitter comments
- User selects the column containing tweet text
- Text is cleaned and preprocessed
- Sentiment is predicted using the trained model
- App displays:
  - Overall sentiment distribution
  - Dominant sentiment (Positive / Neutral / Negative)
  - Word Cloud visualization


🛠️ Technologies Used

- Programming Language: Python
- NLP: NLTK
- Machine Learning: Scikit-learn
- Vectorization: TF-IDF
- Visualization: WordCloud, Matplotlib
- Web Framework: Streamlit

⚠️ Disclaimer

This project is developed strictly for academic purposes. The sentiment predictions are based on a machine learning model and may not always be 100% accurate.
