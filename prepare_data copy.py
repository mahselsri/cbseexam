import os
import json
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths to JSON files
JSON_FILES = {
    "CHEMISTRY": "pdfs/CHEMISTRY/chemistry_consolidated.json",
    "PHYSICS": "pdfs/PHYSICS/physics_consolidated.json"  # add later when ready
}

# Output
VECTOR_INDEX_FILE = "vector.index"
QUESTIONS_FILE = "questions.pkl"
QUESTION_TEXT_FILE = "questions.txt"
METADATA_FILE = "questions_metadata.pkl"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

all_questions = []
question_tags = []
metadata_list = []

for subject, file_path in JSON_FILES.items():
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    papers = full_data.get("papers", [])
    for paper in papers:
        year = paper.get("year", "Unknown")
        filename = paper.get("filename", "")
        questions = paper.get("questions", [])

        for q in questions:
            question_text = q.get("question", "").strip()
            if not question_text or len(question_text) < 10:
                continue

            # Add MCQ choices if present
            if q.get("choices"):
                choices = q["choices"]
                question_text += " Choices: " + "; ".join(choices)

            tagged_text = f"[{subject}] {question_text}"
            all_questions.append(question_text)
            question_tags.append(tagged_text)

            metadata_list.append({
                "subject": subject,
                "year": year,
                "file": filename,
                "chapter": q.get("chapter"),
                "marks": q.get("marks"),
                "original": question_text
            })

print(f"\n✅ Total questions collected: {len(all_questions)}")

# Save raw question text
with open(QUESTION_TEXT_FILE, "w", encoding="utf-8") as f:
    for q in question_tags:
        f.write(q + "\n")

# Save for use in Streamlit app
with open(QUESTIONS_FILE, "wb") as f:
    pickle.dump(question_tags, f)

with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadata_list, f)

# Generate embeddings
print("🔍 Generating embeddings...")
embeddings = model.encode(all_questions)
embeddings = np.array(embeddings).astype("float32")

# Create and store FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, VECTOR_INDEX_FILE)

print(f"\n✅ Vector index saved successfully for {len(all_questions)} questions.")
