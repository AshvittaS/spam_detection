from joblib import dump
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from src.preprocess import load_dataset
from src.evaluate import compare_models


def train_models():

    # Set experiment name
    mlflow.set_experiment("Spam Detection")

    # Load dataset
    df = load_dataset()

    X = df["clean_message"]
    y = df["label"]

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Models
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(random_state=42),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            random_state=42
        )
    }

    trained_models = {}

    # Train all models
    for name, model in models.items():

        print(f"Training {name}...")

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            trained_models[name] = model

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Parameters
            mlflow.log_param("Model", name)
            mlflow.log_param("Vectorizer", "TF-IDF")
            mlflow.log_param("Max Features", 5000)

            # Metrics
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1 Score", f1)

            # Save model to MLflow
            mlflow.sklearn.log_model(model, name="model")

    # Compare models
    results = compare_models(trained_models, X_test, y_test)
    print(results)

    # Save vectorizer
    dump(tfidf, "models/tfidf_vectorizer.pkl")

    # Save models
    dump(trained_models["Naive Bayes"], "models/naive_bayes.pkl")
    dump(trained_models["SVM"], "models/svm.pkl")
    dump(trained_models["XGBoost"], "models/xgboost.pkl")
    dump(trained_models["AdaBoost"], "models/adaboost.pkl")

    print("\nModels saved successfully!")

    return results


if __name__ == "__main__":
    train_models()