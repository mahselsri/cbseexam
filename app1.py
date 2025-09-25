import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load Gemini
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-pro")

# Load data
questions = pickle.load(open("questions.pkl", "rb"))
index = faiss.read_index("vector.index")

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.title("📘 CBSE Exam Helper - Physics & Chemistry")
st.markdown("Search past CBSE questions and get AI-powered explanations using Gemini.")

query = st.text_input("🔍 Enter your topic or question (e.g. 'Ohm's Law', 'Explain Fleming's Rule'):")

if query:
    q_embed = embedder.encode([query])
    D, I = index.search(q_embed, k=3)

    st.subheader("📄 Similar Questions:")
    for i in I[0]:
        question = questions[i]
        st.markdown(f"- {question}")

    if st.button("💡 Explain Best Match"):
        prompt = f"Explain this CBSE 12th question in simple terms for a student: {questions[I[0][0]]}"
        response = model_gemini.generate_content(prompt)
        st.subheader("✍️ Gemini's Explanation:")
        st.write(response.text)
