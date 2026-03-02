"""
Training script for the Career Guidance Chatbot NLP Model.

Steps:
1. Load intents.json
2. Preprocess all patterns (tokenize, lemmatize)
3. Vectorize using TF-IDF (Term Frequency-Inverse Document Frequency)
4. Encode labels using LabelEncoder
5. Build a Multi-Layer Perceptron (MLP) neural network classifier
6. Train and save the model, vectorizer, and label encoder

Tech: Python, scikit-learn (MLPClassifier neural network), NLTK, NLP
"""

import json
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from utils import preprocess_text

# ──────────────────────────────────────────────
# 1. Load Intents Data
# ──────────────────────────────────────────────
print("=" * 55)
print("  🎯 Career Guidance Chatbot – Model Training")
print("=" * 55)

print("\n📂 Step 1: Loading intents data...")

with open('intents.json', 'r', encoding='utf-8') as f:
    intents_data = json.load(f)

print(f"   ✅ Loaded {len(intents_data['intents'])} intent categories")

# ──────────────────────────────────────────────
# 2. Prepare Training Data
# ──────────────────────────────────────────────
print("\n📝 Step 2: Preprocessing training patterns...")

texts = []      # preprocessed input sentences
labels = []     # corresponding intent tags

for intent in intents_data['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        processed = preprocess_text(pattern)
        texts.append(processed)
        labels.append(tag)

print(f"   Total training samples: {len(texts)}")
print(f"   Total unique intents:   {len(set(labels))}")
print(f"   Intent tags: {sorted(set(labels))}")

# ──────────────────────────────────────────────
# 3. TF-IDF Vectorization
# ──────────────────────────────────────────────
print("\n🔤 Step 3: Vectorizing text using TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),  # unigrams and bigrams
    sublinear_tf=True
)
X = vectorizer.fit_transform(texts)

print(f"   Feature matrix shape: {X.shape}")
print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"   Sample features: {list(vectorizer.vocabulary_.keys())[:10]}...")

# ──────────────────────────────────────────────
# 4. Encode Labels
# ──────────────────────────────────────────────
print("\n🏷️  Step 4: Encoding labels...")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

print(f"   Number of classes: {num_classes}")
print(f"   Classes: {list(label_encoder.classes_)}")

# ──────────────────────────────────────────────
# 5. Build Neural Network (MLP Classifier)
# ──────────────────────────────────────────────
print("\n🧠 Step 5: Building Multi-Layer Perceptron neural network...")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),       # Two hidden layers (like Keras Dense layers)
    activation='relu',                   # ReLU activation function
    solver='adam',                       # Adam optimizer (same as Keras)
    max_iter=500,                        # Maximum training epochs
    early_stopping=True,                 # Early stopping (like Keras EarlyStopping)
    validation_fraction=0.15,            # 15% validation split
    n_iter_no_change=20,                 # Patience for early stopping
    random_state=42,
    verbose=True,
    learning_rate='adaptive',
    alpha=0.001                          # L2 regularization (like Dropout)
)

print("   Architecture: Input → Dense(128, relu) → Dense(64, relu) → Softmax")
print(f"   Optimizer: Adam | Regularization: L2 (alpha=0.001)")
print(f"   Early Stopping: patience=20 | Max epochs: 500")

# ──────────────────────────────────────────────
# 6. Train the Model
# ──────────────────────────────────────────────
print("\n🚀 Step 6: Training the model...")

model.fit(X, y)

train_accuracy = model.score(X, y)
print(f"\n   ✅ Training complete!")
print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
print(f"   Iterations run:    {model.n_iter_}")
print(f"   Final loss:        {model.loss_:.6f}")

# Cross-validation for robustness check
print("\n📊 Step 6b: Cross-validation check...")
cv_scores = cross_val_score(model, X, y, cv=min(5, len(set(labels))), scoring='accuracy')
print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ──────────────────────────────────────────────
# 7. Save All Artifacts
# ──────────────────────────────────────────────
print("\n💾 Step 7: Saving model and artifacts...")

# Save model
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✅ Model saved:         chatbot_model.pkl")

# Save vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("   ✅ TF-IDF Vectorizer:   vectorizer.pkl")

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("   ✅ Label encoder:       label_encoder.pkl")

# ──────────────────────────────────────────────
# 8. Quick Test
# ──────────────────────────────────────────────
print("\n🧪 Step 8: Quick smoke test...")

test_inputs = [
    "Hello",
    "Tell me about IT careers",
    "How to write a resume",
    "I want to switch careers",
    "What skills should I learn",
    "Bye"
]

for test in test_inputs:
    processed = preprocess_text(test)
    features = vectorizer.transform([processed])
    probs = model.predict_proba(features)[0]
    pred_idx = np.argmax(probs)
    tag = label_encoder.inverse_transform([pred_idx])[0]
    conf = probs[pred_idx]
    status = "✅" if conf > 0.3 else "⚠️"
    print(f"   {status} \"{test}\" → {tag} ({conf:.1%})")

print("\n" + "=" * 55)
print("  🎉 All done! Model is ready for the chatbot.")
print("  Run: python -m streamlit run app.py")
print("=" * 55)
