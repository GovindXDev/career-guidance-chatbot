# 🎯 Career Guidance AI Chatbot

An NLP-based chatbot that provides intelligent career guidance through an interactive web interface. The chatbot uses a neural network for intent classification and NLTK for text preprocessing.

## 🚀 Features

- **17 Career Intent Categories** – IT, Medicine, Engineering, Business, Arts, and more
- **NLP Pipeline** – Text preprocessing with tokenization and lemmatization
- **Neural Network** – MLP Classifier for accurate intent classification
- **TF-IDF Vectorization** – Converts text to meaningful numerical features
- **Interactive Chat UI** – Beautiful Streamlit interface with dark theme
- **Quick Questions** – Sidebar with common career questions
- **Confidence Scoring** – Shows prediction confidence for transparency

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python** | Core programming language |
| **scikit-learn** | MLPClassifier neural network for intent classification |
| **NLTK** | Natural Language Processing (tokenization, lemmatization) |
| **Streamlit** | Web-based chat interface |
| **NumPy** | Numerical operations |

## 📂 Project Structure

```
career-guidance-chatbot/
├── intents.json        # Training data – 17 intents with patterns & responses
├── utils.py            # NLP utilities – preprocessing, prediction, response
├── train_model.py      # Model training script – TF-IDF + MLP neural network
├── app.py              # Streamlit web application
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/career-guidance-chatbot.git
cd career-guidance-chatbot
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train_model.py
```

This will generate:
- `chatbot_model.pkl` – Trained neural network
- `vectorizer.pkl` – Fitted TF-IDF vectorizer
- `label_encoder.pkl` – Fitted label encoder

### 5. Launch the Chatbot

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## 🧠 How It Works

```
User Input → Preprocessing (NLTK) → TF-IDF Vectorization → MLP Neural Network → Intent → Response
```

1. **User types a message** in the Streamlit chat interface
2. **Text Preprocessing** – Lowercase, remove special chars, tokenize, lemmatize
3. **TF-IDF Vectorization** – Convert text to numerical feature vector
4. **Intent Classification** – Neural network predicts the most likely intent
5. **Response Generation** – Random response selected from the predicted intent
6. **Display** – Response shown in chat with intent tag and confidence score

## 💬 Supported Topics

| Category | Example Questions |
|----------|-------------------|
| 🖥️ IT Careers | "Tell me about IT careers", "How to become a developer" |
| 🏥 Medicine | "Medical career options", "How to become a doctor" |
| ⚙️ Engineering | "Best engineering fields", "Mechanical engineering career" |
| 💼 Business | "MBA career options", "Marketing career" |
| 🎨 Arts | "Creative career options", "Design career" |
| 📝 Resume Tips | "How to write a resume", "Resume format" |
| 🎤 Interview Prep | "Interview tips", "Common interview questions" |
| 📚 Education | "What should I study", "Best courses to take" |
| 💡 Skills | "What skills should I learn", "Skills for career growth" |
| 🔄 Career Switch | "How to switch careers", "Career transition advice" |
| 💰 Salary Info | "Average salary", "Highest paying jobs" |
| 🏠 Freelancing | "How to start freelancing", "Remote work options" |

## 📊 Model Performance

| Test Query | Predicted Intent | Confidence |
|-----------|-----------------|------------|
| "Hello" | greeting | 86.7% |
| "Tell me about IT careers" | career_IT | 94.1% |
| "How to write a resume" | resume_tips | 95.8% |
| "I want to switch careers" | career_switch | 91.9% |
| "How to start freelancing" | freelancing | 95.9% |

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
