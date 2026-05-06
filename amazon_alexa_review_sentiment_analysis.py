# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords', quiet=True)

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from wordcloud import WordCloud

import pickle
import re

# ========================
# LOAD DATA (FIXED)
# ========================
data = pd.read_csv("amazon_alexa.tsv", delimiter="\t", quoting=3)

data.dropna(inplace=True)
data['length'] = data['verified_reviews'].apply(len)

# ========================
# PREPROCESSING
# ========================
corpus = []
stemmer = PorterStemmer()

for i in range(data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['verified_reviews'])
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = ' '.join(review)
    corpus.append(review)

# ========================
# VECTORIZER
# ========================
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values

pickle.dump(cv, open('countVectorizer.pkl', 'wb'))

# ========================
# TRAIN TEST SPLIT
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15
)

# ========================
# SCALING
# ========================
scaler = MinMaxScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

pickle.dump(scaler, open('scaler.pkl', 'wb'))

# ========================
# RANDOM FOREST
# ========================
model_rf = RandomForestClassifier()
model_rf.fit(X_train_scl, y_train)

print("RF Train Accuracy:", model_rf.score(X_train_scl, y_train))
print("RF Test Accuracy:", model_rf.score(X_test_scl, y_test))

y_preds = model_rf.predict(X_test_scl)

# ========================
# XGBOOST
# ========================
model_xgb = XGBClassifier()
model_xgb.fit(X_train_scl, y_train)

print("XGB Train Accuracy:", model_xgb.score(X_train_scl, y_train))
print("XGB Test Accuracy:", model_xgb.score(X_test_scl, y_test))

# ❌ FIXED BUG HERE
y_preds = model_xgb.predict(X_test_scl)

cm = confusion_matrix(y_test, y_preds)
ConfusionMatrixDisplay(cm).plot()
plt.show()

# ========================
# SAVE MODEL
# ========================
pickle.dump(model_xgb, open('model_xgb.pkl', 'wb'))

# ========================
# DECISION TREE
# ========================
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_scl, y_train)

print("DT Train Accuracy:", model_dt.score(X_train_scl, y_train))
print("DT Test Accuracy:", model_dt.score(X_test_scl, y_test))

#Confusion Matrix
cm = confusion_matrix(y_test, y_preds)
print(cm)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_dt.classes_)
cm_display.plot()
plt.show()

