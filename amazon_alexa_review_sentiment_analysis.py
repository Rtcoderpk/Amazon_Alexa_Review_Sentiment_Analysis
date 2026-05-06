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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# =========================
# INIT
# =========================

STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

os.makedirs("Models", exist_ok=True)

# =========================
# CLEAN TEXT FUNCTION (IMPORTANT FIX)
# =========================

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS and len(word) > 2]
    return ' '.join(text)

# =========================
# STREAMLIT UI
# =========================

st.title("📊 Amazon Alexa Sentiment Analysis")

# =========================
# LOAD DATA (FIXED PATH)
# =========================

data = pd.read_csv("amazon_alexa.tsv", delimiter="\t", quoting=3)
data.dropna(inplace=True)

# =========================
# PREPROCESS DATA
# =========================

corpus = []

for i in range(len(data)):
    review = clean_text(data.iloc[i]['verified_reviews'])
    corpus.append(review)

# =========================
# VECTORIZATION
# =========================

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values

# SAVE VECTOR
pickle.dump(cv, open('Models/countVectorizer.pkl', 'wb'))

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15
)

# =========================
# SCALING
# =========================

scaler = MinMaxScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

pickle.dump(scaler, open('Models/scaler.pkl', 'wb'))

# =========================
# MODEL TRAINING
# =========================

model = XGBClassifier(
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train_scl, y_train)

# =========================
# ACCURACY
# =========================

train_acc = model.score(X_train_scl, y_train)
test_acc = model.score(X_test_scl, y_test)

st.subheader("Model Accuracy")
st.write("Train Accuracy:", train_acc)
st.write("Test Accuracy:", test_acc)

# =========================
# USER INPUT
# =========================

st.subheader("🧪 Test Your Review")

user_input = st.text_input("Enter review")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a review")

    else:
        cleaned = clean_text(user_input)

        vector = cv.transform([cleaned]).toarray()
        vector = scaler.transform(vector)

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        confidence = prob[1]

        # =========================
        # FIXED DECISION LOGIC
        # =========================

        if confidence >= 0.60:
            st.success(f"Positive Review 👍 (Confidence: {confidence:.2f})")
        else:
            st.error(f"Negative Review 👎 (Confidence: {confidence:.2f})")
