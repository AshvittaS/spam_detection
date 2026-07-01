# Spam or Not Spam
Deployed link :[link](https://huggingface.co/spaces/Ashvitta07/Spam_or_not_Spam)
This project is a complete spam detection application that turns SMS-style text into a real-time prediction experience. It combines machine learning, experiment tracking, and a simple web interface into one polished workflow.

## What makes this project unique

### 1. End-to-end spam detection pipeline
This is not just a notebook experiment. The project includes a full workflow from data cleaning to model training, evaluation, saving models, and prediction through a live app.

### 2. Multiple models compared side by side
I trained and compared several classifiers instead of relying on a single model:
- Naive Bayes
- SVM
- XGBoost
- AdaBoost

This makes the project more practical and gives it a stronger machine learning experimentation feel.

### 3. MLflow experiment tracking
The training process is connected with MLflow, so model runs are tracked with:
- model name
- vectorizer settings
- feature count
- accuracy
- precision
- recall
- F1 score

That adds a professional experiment-management layer to the project.

### 4. Custom text preprocessing pipeline
The project uses a dedicated cleaning step for message text, including:
- lowercasing
- removing URLs
- removing numbers
- removing punctuation
- removing stop words

This makes the model input cleaner and more reliable for real-world text.

### 5. Interactive Gradio interface
A simple but effective web app was built using Gradio, allowing users to type a message and instantly get a spam or not-spam prediction.

### 6. Model artifacts are saved for reuse
The project saves trained components such as:
- TF-IDF vectorizer
- trained classifiers

This means the system is ready to be reused beyond the training step.

### 7. Deployed as a Hugging Face Space
The app is also published as a Hugging Face Space, which makes it easy to share and use online.

## Project highlights
- Practical ML project with real-world use case
- Clean preprocessing pipeline for text data
- Comparison of multiple classification models
- Reproducible training and evaluation setup
- User-friendly interface for predictions
- Deployment-ready architecture

## Summary
This project stands out because it combines machine learning experimentation, model tracking, deployment, and a simple user-facing app into one complete solution for spam detection.
