import pandas as pd
import re
import string

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

def load_dataset(path="dataset/spam.csv"):
    df = pd.read_csv(path, encoding="latin-1")

    # Keep only required columns
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

    # Remove missing values and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Convert labels
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Clean messages
    df["clean_message"] = df["message"].apply(clean_text)

    return df