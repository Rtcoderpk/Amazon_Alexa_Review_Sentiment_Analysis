import os
import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# INIT
# =========================

STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

os.makedirs("Models", exist_ok=True)

# =========================
# CLEAN TEXT FUNCTION
# =========================

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS and len(word) > 2]
    return ' '.join(text)

# =========================
# STREAMLIT UI
# =========================

st.title("📊 Amazon Alexa Sentiment Analysis (Improved Model)")

# =========================
# LOAD DATA
# =========================

data = pd.read_csv("amazon_alexa.tsv", delimiter="\t", quoting=3)
data.dropna(inplace=True)

# =========================
# PREPROCESS
# =========================

corpus = [clean_text(text) for text in data['verified_reviews']]

# =========================
# TF-IDF VECTOR (BETTER THAN COUNT VECTORIZER)
# =========================

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(corpus).toarray()
y = data['feedback'].values

pickle.dump(tfidf, open("Models/tfidf.pkl", "wb"))

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# =========================
# MODEL (BEST FOR SENTIMENT)
# =========================

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'  # IMPORTANT FIX
)

model.fit(X_train, y_train)

# =========================
# ACCURACY
# =========================

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

st.subheader("Model Accuracy")
st.write("Train Accuracy:", train_acc)
st.write("Test Accuracy:", test_acc)

# =========================
# SAVE MODEL
# =========================

pickle.dump(model, open("Models/model.pkl", "wb"))

# =========================
# USER INPUT PREDICTION
# =========================

st.subheader("🧪 Test Your Review")

user_input = st.text_input("Enter review")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a review")

    else:

        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned]).toarray()

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        confidence = max(prob)

        if prediction == 1:
            st.success(f"Positive Review 👍 ({confidence:.2f})")
        else:
            st.error(f"Negative Review 👎 ({confidence:.2f})")
