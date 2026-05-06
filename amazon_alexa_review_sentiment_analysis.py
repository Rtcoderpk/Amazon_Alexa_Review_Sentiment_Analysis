import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import streamlit as st
import os
import pickle
import re

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

st.title("Amazon Alexa Sentiment Analysis")

# -------------------------------
# FIX 1: Correct dataset path
# -------------------------------
DATA_PATH = "amazon_alexa.tsv"

if not os.path.exists(DATA_PATH):
    st.error("Dataset file not found. Please upload amazon_alexa.tsv in repo root.")
    st.stop()

data = pd.read_csv(DATA_PATH, delimiter="\t", quoting=3)

data.dropna(inplace=True)
data['length'] = data['verified_reviews'].apply(len)

# -------------------------------
# Preprocessing
# -------------------------------
corpus = []
stemmer = PorterStemmer()

for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['verified_reviews'])
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    corpus.append(" ".join(review))

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values

# -------------------------------
# FIX 2: Create folder if not exists
# -------------------------------
os.makedirs("Models", exist_ok=True)

pickle.dump(cv, open("Models/countVectorizer.pkl", "wb"))

# Scaling
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pickle.dump(scaler, open("Models/scaler.pkl", "wb"))

# -------------------------------
# Model
# -------------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

st.subheader("Model Accuracy")
st.write("Train:", model.score(X_train, y_train))
st.write("Test:", model.score(X_test, y_test))

# -------------------------------
# Prediction UI
# -------------------------------
st.subheader("Test Your Review")

user_input = st.text_area("Enter review")

if st.button("Predict"):
    review = re.sub('[^a-zA-Z]', ' ', user_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)

    input_data = cv.transform([review]).toarray()
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Positive Review 👍")
    else:
        st.error("Negative Review 👎")
