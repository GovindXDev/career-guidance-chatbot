"""
Utility functions for the Career Guidance Chatbot.
Handles text preprocessing, intent prediction, and response generation.
"""

import json
import random
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Preprocess input text:
    1. Convert to lowercase
    2. Remove special characters
    3. Tokenize
    4. Lemmatize each token
    Returns the cleaned, lemmatized text as a string.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)


def load_intents(filepath='intents.json'):
    """Load intents from the JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def predict_intent(text, model, vectorizer, label_encoder):
    """
    Predict the intent of the given text using the trained model.
    Returns (predicted_tag, confidence_score).
    """
    # Preprocess
    processed = preprocess_text(text)

    # Vectorize using fitted TF-IDF vectorizer
    features = vectorizer.transform([processed])

    # Predict probabilities
    probabilities = model.predict_proba(features)[0]
    predicted_index = np.argmax(probabilities)
    confidence = float(probabilities[predicted_index])

    # Decode label
    tag = label_encoder.inverse_transform([predicted_index])[0]

    return tag, confidence


def get_response(tag, intents_data, confidence, threshold=0.3):
    """
    Get a random response for the predicted intent tag.
    If confidence is below threshold, return a fallback response.
    """
    if confidence < threshold:
        return (
            "I'm not quite sure I understand. Could you rephrase your question? "
            "You can ask me about careers in IT, medicine, engineering, business, arts, "
            "resume tips, interview preparation, skills development, and more!"
        )

    for intent in intents_data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "I'm sorry, I couldn't find an answer for that. Try asking about specific career fields!"

