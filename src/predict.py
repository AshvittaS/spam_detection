from joblib import load
from src.preprocess import clean_text

# Load the saved model and vectorizer once
model = load("models/svm.pkl")          # or naive_bayes.pkl, xgboost.pkl, etc.
vectorizer = load("models/tfidf_vectorizer.pkl")


def predict_spam(message):
    """
    Predict whether a message is Spam or Not Spam.
    """

    # Clean the input text
    cleaned = clean_text(message)

    # Convert to TF-IDF features
    vector = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "🚨 Spam"
    else:
        return "✅ Not Spam"