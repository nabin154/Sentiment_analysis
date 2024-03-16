import pickle
from flask import Flask, render_template, request
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

model_file_path = 'sentimentenglish_model.pkl'
vectorizer_file_path = 'vectorizerenglish.pkl'

# Check if files exist before loading
if os.path.exists(model_file_path) and os.path.exists(vectorizer_file_path):
    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(vectorizer_file_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
else:
    raise FileNotFoundError(f"Files {model_file_path} or {vectorizer_file_path} not found.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']

        # Vectorize the input text
        text_vectorized = vectorizer.transform([text])

        # Use the pre-trained sentiment analysis model to predict sentiment
        predicted_sentiment = model.predict(text_vectorized)

        # Map the numerical prediction to sentiment labels
        sentiment = 'Positive' if predicted_sentiment == 0 else 'Negative' if predicted_sentiment == 1 else 'Neutral'

        return render_template('index.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)