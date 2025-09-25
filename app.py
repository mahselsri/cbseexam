import streamlit as st
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from dotenv import load_dotenv
import hashlib
# CONFIGURE GEMINI
import os
import re
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load Gemini
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# Load vector index and question data
index = faiss.read_index("vector.index")
with open("questions.pkl", "rb") as f:
    questions = pickle.load(f)
with open("questions_metadata.pkl", "rb") as f:
    metadata_list = pickle.load(f)

# Load model for embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

import re
import re

def auto_latex_format(text):
    if not isinstance(text, str):
        return text

    # Convert chemical subscripts like H2O → H$_2$O
    text = re.sub(r'([A-Za-z])(\d+)', r'\1$_\2$', text)

    # Convert math superscripts like x^2 → $x^2$
    text = re.sub(r'([a-zA-Z0-9])\^(\d+)', r'$\1^\2$', text)

    # Arrow replacements
    text = text.replace("->", r"$\rightarrow$")
    text = text.replace("→", r"$\rightarrow$")

    return text

def render_question_with_latex(text):
    text = auto_latex_format(text)  # ← format before rendering
   
    # If text contains $$...$$ or $...$, render separately
    # Example: E = mc^2 → $E = mc^2$
    latex_exprs = re.findall(r"\$(.+?)\$", text)
    
    if latex_exprs:
        for expr in latex_exprs:
            text = text.replace(f"${expr}$", "")
            st.latex(expr)
    st.markdown(text)

# Helper: Search
def search_questions(query, top_k=5):
    query_embedding = model.encode([query]).astype("float32")
    D, I = index.search(query_embedding, top_k)
    results = []
    for i in I[0]:
        if i < len(questions):
            results.append({
                "question": questions[i],
                **metadata_list[i]
            })
    return results

# Helper: Generate guidebook per subject
@st.cache_data
def generate_guidebook(subject):
    df = pd.DataFrame(metadata_list)
    df = df[df["subject"] == subject.upper()]
    df["formatted"] = df.apply(lambda row: f"[{row['year']}] ({row['marks']}m) {row['original']}", axis=1)
    return df

# Helper: Plot chapter-wise insights
def plot_chapter_insights(subject):
    df = pd.DataFrame(metadata_list)
    df = df[df["subject"] == subject.upper()]
    if "chapter" not in df.columns:
        st.warning("Chapter information not available in metadata.")
        return
    mark_dist = df.groupby("chapter")["marks"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=mark_dist.values, y=mark_dist.index, ax=ax)
    ax.set_title(f"{subject} Chapter-wise Mark Distribution")
    ax.set_xlabel("Total Marks")
    st.pyplot(fig)

# Helper: Gemini Answer Generator
def get_gemini_answer(question_text):
    try:
        prompt = f""" You are a CBSE 12th subject expert. Answer the following question in clear, simple markdown.Use LaTeX notation for any formulas, written like `$H_2SO_4$` or `$v = u + at$`.Question: \n{question_text}"""

        #prompt = f"Please provide a detailed, CBSE Class 12 level answer for the following question:\n{question_text}"
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {e}"

# Streamlit UI
st.set_page_config(page_title="CBSE Smart Revision", layout="wide")
st.title("📘 CBSE Class 12 AI-powered Revision Assistant")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Ask a Question", "📚 Guidebook", "📊 Insights"])

with tab1:
    st.subheader("🔍 Search from past questions")
    query = st.text_input("Enter your topic or question")
    top_k = st.slider("How many results to show?", 1, 20, 5)
    if st.button("Search") and query:
        results = search_questions(query, top_k=top_k)
        for idx, res in enumerate(results):
            st.markdown(f"**{res['question']}**")
            st.caption(f"{res['subject']} | {res['year']} | Sec {res['section']} | Marks: {res['marks']}")

            toggle_key = f"toggle_{res['year']}_{res['section']}_{res['marks']}_{idx}"

            if st.toggle(f"🔍 Show AI Answer", key=toggle_key):
                with st.spinner("Getting answer from Gemini..."):
                    answer = get_gemini_answer(res['question'])
                    st.success("Answer:")
                    st.markdown(answer)


with tab2:
    st.subheader("📚 Chapter-wise Interactive Guidebook")
    subject = st.selectbox("Choose Subject", ["CHEMISTRY", "PHYSICS"])
    guide = generate_guidebook(subject)

    years = sorted(set(guide["year"]))
    chapters = sorted(set(guide["chapter"]))
    marks_set = sorted(set(guide["marks"]))

    st.markdown("### 🔎 Filter Questions")
    filter_years = st.multiselect("Year", years, default=years)
    filter_chapters = st.multiselect("Chapter", chapters, default=chapters)
    filter_marks = st.multiselect("Marks", marks_set, default=marks_set)

    filtered = guide[
        (guide["year"].isin(filter_years)) &
        (guide["chapter"].isin(filter_chapters)) &
        (guide["marks"].isin(filter_marks))
    ]

    for chapter in sorted(filtered["chapter"].unique()):
        with st.expander(f"📘 {chapter}"):
            chapter_df = filtered[filtered["chapter"] == chapter]
            for mark in sorted(chapter_df["marks"].unique()):
                st.markdown(f"#### 📝 {mark}-Mark Questions")
                for _, row in chapter_df[chapter_df["marks"] == mark].iterrows():
                    #st.markdown(f"**• {row['year']}** — {row['original']}")
                    st.markdown(f"**• {row['year']}**")
                    render_question_with_latex(row["original"])

                    # Generate a unique key using hash
                    unique_id = hashlib.md5(f"{row['original']}_{row['year']}_{row['marks']}".encode()).hexdigest()
                    if st.button(f"AI Answer ➤ {row['year']} | {row['marks']}m", key=f"answer_btn_{row.name}"):
                        with st.spinner("Gemini answering..."):
                            answer = get_gemini_answer(row['original'])
                            st.markdown(f"**Answer:**\n{answer}")

with tab3:
    st.subheader("📊 Subject-wise Preparation Insights")
    subject = st.selectbox("Choose Subject for Insights", ["CHEMISTRY", "PHYSICS"], key="insights")
    plot_chapter_insights(subject)
    st.info("This graph helps your daughter focus on high-weightage chapters.")
