import os
import json
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# JSON files per subject
JSON_FILES = {
    "CHEMISTRY": "pdfs/CHEMISTRY/chemistry_chapters_updated.json",
    "PHYSICS": "pdfs/PHYSICS/physics_chapters_updated.json"  # add later when ready
}

# Output files
VECTOR_INDEX_FILE = "vector.index"
QUESTIONS_FILE = "questions.pkl"
QUESTION_TEXT_FILE = "questions.txt"
METADATA_FILE = "questions_metadata.pkl"

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

all_questions = []
question_tags = []
metadata_list = []

for subject, file_path in JSON_FILES.items():
    if not os.path.exists(file_path):
        print(f"⚠️ Skipping missing file: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    for paper in full_data.get("papers", []):
        year = paper.get("year", "Unknown")
        filename = paper.get("filename", "Unknown")
        questions = paper.get("questions", [])

        if not questions:
            continue  # Skip papers with no questions

        for q in questions:
            q_text = q.get("text", "").strip()
            if not q_text or len(q_text) < 10:
                continue
            q_chapter = q.get("chapter", "")
            q_mark = q.get("marks", None)
            q_section = q.get("section", "")
            q_num = q.get("number", "")
            choices = q.get("choices", {})

            # Append choices if MCQ-style
            if isinstance(choices, dict) and choices:
                choices_text = " Choices: " + "; ".join([f"{k}) {v}" for k, v in choices.items()])
                q_text += choices_text

            full_text = f"[{subject}][{year}][Sec {q_section}] Q{q_num}: {q_text}"
            all_questions.append(q_text)
            question_tags.append(full_text)

            metadata_list.append({
                "subject": subject,
                "year": year,
                "chapter":q_chapter,
                "section": q_section,
                "marks": q_mark,
                "filename": filename,
                "original": q_text
            })

print(f"\n✅ Total valid questions: {len(all_questions)}")

# Save raw text for reference
with open(QUESTION_TEXT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(question_tags))

# Save for app usage
with open(QUESTIONS_FILE, "wb") as f:
    pickle.dump(question_tags, f)

with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadata_list, f)

# Embedding + FAISS
if not all_questions:
    print("❌ No questions to embed. Exiting.")
    exit()

print("🔍 Encoding questions...")
embeddings = model.encode(all_questions)
embeddings = np.array(embeddings).astype("float32")

if embeddings.ndim != 2 or embeddings.shape[0] == 0:
    print(f"❌ Invalid embeddings: shape={embeddings.shape}")
    exit()

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, VECTOR_INDEX_FILE)

print(f"\n✅ Vector index created with {embeddings.shape[0]} questions.")
