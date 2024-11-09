from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import numpy as np

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET", "POST"])
def landing_page():
    sentiment = None
    if request.method == "POST":
        input_sent = request.form.get("text", "").strip()
        if input_sent:  # Check if input text is not empty
            sentiment = predict(input_sent)
        else:
            sentiment = "Please enter some text to analyze." 
    return render_template("landing_page.html", content=sentiment)


# Prediction function
def predict(input_text):
    # Load models and pre-processing objects
    try:
        with open("Models/model_xgb.pkl", "rb") as f:
            predictor = pickle.load(f)
        with open("Models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("Models/countVectorizer.pkl", "rb") as f:
            cv = pickle.load(f)
    except FileNotFoundError:
        return "Error loading model or pre-processing files."

    # Run prediction and return sentiment
    return prediction(predictor, scaler, cv, input_text)

# Prediction processing function
def prediction(predictor, scaler, cv, text_input):
    # Preprocess input text
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    processed_review = " ".join(review)
    corpus.append(processed_review)
    
    # Transform input text
    X = cv.transform(corpus).toarray()
    X_scaled = scaler.transform(X)
    
    # Predict sentiment
    try:
        prediction_probs = predictor.predict_proba(X_scaled)
        prediction_label = np.argmax(prediction_probs, axis=1)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction."

    return "POSITIVE" if prediction_label == 1 else "NEGATIVE"

if __name__ == "__main__":
    app.run(port=5001, debug=True)
