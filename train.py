import pandas as pd

# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0   # Fake news
true["label"] = 1   # Real news

# Combine both datasets
df = pd.concat([fake, true])

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

import re

# Clean text function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove special characters
    text = text.lower()  # convert to lowercase
    return text

# Apply cleaning on text column
df["text"] = df["text"].apply(clean_text)

from sklearn.feature_extraction.text import TfidfVectorizer

# Features and labels
X = df["text"]
y = df["label"]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform text into numbers
X_vectorized = vectorizer.fit_transform(X)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

import joblib

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")