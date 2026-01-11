import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Custom CSS for Arnav Srivastava branding
st.markdown("""
<style>
.main-header {font-size: 3rem; color: #1f77b4; font-weight: bold;}
.arnav-credit {font-size: 0.9rem; color: #666; margin-top: 1rem;}
.sidebar .sidebar-content {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
</style>
""", unsafe_allow_html=True)

# Load model (cached for performance)
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, 
                   device=0 if torch.cuda.is_available() else -1)
    return pipe

# Page config
st.set_page_config(page_title="Gemini AI by Arnav Srivastava", layout="wide")

# Header
st.markdown('<h1 class="main-header">🤖 Gemini-Like AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="arnav-credit">Created by <strong>Arnav Srivastava</strong> | Rourkela, Odisha | Class 12 JEE Aspirant & AI Developer</p>', unsafe_allow_html=True)

# Sidebar with Arnav's profile
with st.sidebar:
    st.markdown("## 👨‍💻 About Arnav")
    st.markdown("""
    - **Class 12 Student** (BITSAT/JEE Mains prep)
    - **Skills**: Python, Java, AI/ML, Full-Stack
    - **Interests**: Competitive Programming, Astronomy
    - **GitHub**: [Your GitHub Link]
    """)
    st.markdown("---")
    st.info("💡 Try: 'Write Python calculator code'")

# Main content
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("💬 Chat with AI")
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything... (code, math, explanations)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Arnav's AI is thinking..."):
                generator = load_model()
                response = generator(prompt, max_length=512, num_return_sequences=1, 
                                   temperature=0.7, do_sample=True)[0]['generated_text']
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.subheader("🚀 Features")
    st.markdown("""
    - **Code Generation** (Python, Java, etc.)
    - **Math Solver** (JEE-level problems)
    - **Physics Explanations**
    - **Chat Memory**
    - **Mobile Optimized**
    """)

# Footer
st.markdown("---")
st.markdown('<p class="arnav-credit">© 2026 Arnav Srivastava | Made with ❤️ for JEE prep & AI exploration</p>', unsafe_allow_html=True)
