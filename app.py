"""
Career Guidance AI Chatbot – Streamlit Application

This is the main interface for the chatbot. It loads the trained NLP model
and provides a beautiful chat interface for career guidance conversations.
"""

import streamlit as st
import pickle
import os
from utils import predict_intent, get_response, load_intents

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Career Guidance AI Chatbot",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }

    .main-header h1 {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #b8c5e8;
        font-size: 1rem;
        margin: 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    }

    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #667eea !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #b8c5e8 !important;
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        border-radius: 15px !important;
        margin-bottom: 0.8rem !important;
        border: 1px solid rgba(102, 126, 234, 0.15) !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Chat input styling */
    [data-testid="stChatInput"] textarea {
        background: rgba(30, 30, 60, 0.8) !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
        border-radius: 15px !important;
        color: white !important;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }

    .feature-card .emoji {
        font-size: 1.3rem;
        margin-right: 0.5rem;
    }

    .feature-card .text {
        color: #d1d5f0;
        font-size: 0.9rem;
    }

    /* Status indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(46, 213, 115, 0.15);
        border: 1px solid rgba(46, 213, 115, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #2ed573;
        margin-top: 0.5rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #2ed573;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Quick action buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: #d1d5f0 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.4), rgba(118, 75, 162, 0.4)) !important;
        border-color: rgba(102, 126, 234, 0.6) !important;
        transform: translateY(-1px) !important;
    }

    /* Divider */
    hr {
        border-color: rgba(102, 126, 234, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load Model and Artifacts
# ──────────────────────────────────────────────
@st.cache_resource
def load_chatbot_resources():
    """Load all ML artifacts (cached so they load only once)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir, 'chatbot_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(base_dir, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(base_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)

    intents_data = load_intents(os.path.join(base_dir, 'intents.json'))

    return model, vectorizer, label_encoder, intents_data


# Check if model exists
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chatbot_model.pkl')
if not os.path.exists(model_path):
    st.error("⚠️ Model not found! Please run `python train_model.py` first to train the chatbot model.")
    st.stop()

model, vectorizer, label_encoder, intents_data = load_chatbot_resources()


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Career Guidance Bot")
    st.markdown('<div class="status-badge"><div class="status-dot"></div>Online & Ready</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 💡 What I Can Help With")

    features = [
        ("💻", "IT & Tech Careers"),
        ("🚀", "Data Science & AI"),
        ("🏥", "Medical Careers"),
        ("⚙️", "Engineering Paths"),
        ("💼", "Business & Management"),
        ("🎨", "Arts & Creative Fields"),
        ("📝", "Resume Building Tips"),
        ("🎤", "Interview Preparation"),
        ("📚", "Education Guidance"),
        ("💡", "Skill Development"),
        ("🔄", "Career Switching"),
        ("💰", "Salary Information"),
        ("🏠", "Freelancing Advice"),
    ]

    for emoji, text in features:
        st.markdown(f'<div class="feature-card"><span class="emoji">{emoji}</span><span class="text">{text}</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ⚡ Quick Questions")
    quick_questions = [
        "What careers are in IT?",
        "How to start in Data Science?",
        "How to write a resume?",
        "Interview preparation tips",
        "I'm confused about my career",
        "What skills should I learn?",
    ]

    for q in quick_questions:
        if st.button(q, key=f"quick_{q}"):
            st.session_state.pending_question = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if 'pending_question' in st.session_state:
            del st.session_state['pending_question']
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#667eea; font-size:0.75rem;'>"
        "Built with ❤️ using Python, scikit-learn & Streamlit"
        "</p>",
        unsafe_allow_html=True
    )


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎯 Career Guidance AI</h1>
    <p>Your intelligent career advisor powered by NLP</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    welcome_msg = (
        "Hello! 👋 I'm your **Career Guidance AI Chatbot**.\n\n"
        "I can help you with:\n"
        "- 🔍 Exploring career paths (IT, Medicine, Engineering, Business, Arts)\n"
        "- 📝 Resume & interview tips\n"
        "- 📚 Education & skill development guidance\n"
        "- 💰 Salary info & freelancing advice\n\n"
        "**Just type your question below to get started!**"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Handle pending quick question from sidebar
if 'pending_question' in st.session_state:
    user_input = st.session_state.pending_question
    del st.session_state['pending_question']

    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get bot response
    tag, confidence = predict_intent(user_input, model, vectorizer, label_encoder)
    response = get_response(tag, intents_data, confidence)

    # Display bot response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
        st.caption(f"🏷️ Intent: `{tag}` | Confidence: `{confidence:.1%}`")
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()

# Chat input
if user_input := st.chat_input("Ask me about careers, skills, interviews, resume tips..."):
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get bot response
    tag, confidence = predict_intent(user_input, model, vectorizer, label_encoder)
    response = get_response(tag, intents_data, confidence)

    # Display bot response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
        st.caption(f"🏷️ Intent: `{tag}` | Confidence: `{confidence:.1%}`")
    st.session_state.messages.append({"role": "assistant", "content": response})
