# ==============================
# AMAZON ALEXA SENTIMENT APP
# FIXED VERSION (DEPLOY READY)
# ==============================

import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ==============================
# STREAMLIT UI
# ==============================

st.title("Amazon Alexa Sentiment Analysis")

# ==============================
# LOAD DATA (FIXED PATH)
# ==============================

data = pd.read_csv("amazon_alexa.tsv", delimiter="\t", quoting=3)

data.dropna(inplace=True)

data['length'] = data['verified_reviews'].apply(len)

# ==============================
# TEXT PREPROCESSING
# ==============================

corpus = []

for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['verified_reviews'])
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = ' '.join(review)
    corpus.append(review)

# ==============================
# BAG OF WORDS
# ==============================

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values

# ==============================
# SAVE MODEL FOLDER (FIX)
# ==============================

os.makedirs("Models", exist_ok=True)

pickle.dump(cv, open('Models/countVectorizer.pkl', 'wb'))

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15
)

# ==============================
# SCALING
# ==============================

scaler = MinMaxScaler()

X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

pickle.dump(scaler, open('Models/scaler.pkl', 'wb'))

# ==============================
# MODEL TRAINING
# ==============================

model = XGBClassifier()
model.fit(X_train_scl, y_train)

# ==============================
# FIXED PREDICTION (IMPORTANT BUG FIX)
# ==============================

y_pred = model.predict(X_test_scl)   # ❗ FIXED (NOT X_test)

acc = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Train Accuracy: {model.score(X_train_scl, y_train)}")
st.write(f"Test Accuracy: {model.score(X_test_scl, y_test)}")

# ==============================
# USER INPUT PREDICTION
# ==============================

st.subheader("Test Your Review")

user_input = st.text_input("Enter review")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a review")
    else:

        # preprocessing same as training
        review = re.sub('[^a-zA-Z]', ' ', user_input)
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = ' '.join(review)

        vector = cv.transform([review]).toarray()
        vector = scaler.transform(vector)

        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("Positive Review 👍")
        else:
            st.error("Negative Review 👎")
