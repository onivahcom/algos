import os, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "models/spam_model.pkl"


def train_and_save_model(): 
    """Train spam detection model if no model.pkl exists"""
    print("Training new spam model...")

    # Load dataset
    data_url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    sms_dataframe = pd.read_csv(data_url, sep="\t", header=None, names=["label", "message"])

    # Features and target
    messages = sms_dataframe["message"]   # X (input: sms text)
    labels = sms_dataframe["label"].map({"ham": 0, "spam": 1})  # y (output: ham/spam)

    # Split dataset
    train_messages, test_messages, train_labels, test_labels = train_test_split(
        messages, labels, test_size=0.2, random_state=42
    )

    # Build pipeline (TF-IDF vectorizer + Naive Bayes classifier)
    spam_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    spam_pipeline.fit(train_messages, train_labels)

    # Evaluate model
    predicted_labels = spam_pipeline.predict(test_messages)
    accuracy = accuracy_score(test_labels, predicted_labels)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(spam_pipeline, file)

    return spam_pipeline


def load_model():
    """Load or train model"""
    if os.path.exists(MODEL_PATH):
        print("Loading existing spam model...")
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)
    else:
        return train_and_save_model()


# Shared model (load once)
spam_model = load_model()


def predict_spam(message_text: str):
    # Predict class (0 = ham, 1 = spam)
    predicted_class = spam_model.predict([message_text])[0]

    # Get probability of being spam
    proba_spam = spam_model.predict_proba([message_text])[0][1]

    return {
        "label": "spam" if predicted_class == 1 else "ham",
        "is_spam": bool(predicted_class),
        "confidence": round(float(proba_spam), 2),  # use spam probability
    }
