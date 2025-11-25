import os
from pdf_reader import extract_pdf_text, save_text
from embedder import chunk_text, embed_and_build_index, save_index, load_index

BASE = r"D:\ADANI CLG\TRI 2\CLASS\GENAI\Assignment\CaseStudySummarizer"

sample_path = BASE + "\\sample_cases"
print("Available PDFs:")
for f in os.listdir(sample_path):
    print(" -", f)

files = [
    "HBR Case Study.pdf",
    "Ferrari 2025 Case.pdf",
    "Tesla Case.pdf"
]

for f in files:
    pdf_path = sample_path + "\\" + f
    print("\n⏳ Extracting text from:", f)

    text = extract_pdf_text(pdf_path)

    out_path = BASE + "\\outputs\\" + f.replace(".pdf", ".txt")
    save_text(out_path, text)

    print("✔ Saved cleaned text →", out_path)

for f in files:
    txt_path = BASE + "\\outputs\\" + f.replace(".pdf", ".txt")
    
    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.read()

    print("\n📌 Processing:", f)

    chunks = chunk_text(text)
    print("✔ Total chunks created:", len(chunks))

    index_path = BASE + "\\outputs\\" + f.replace(".pdf", "_index.pkl")

    db = embed_and_build_index(chunks)
    save_index(db, index_path)

    print("✔ Vector store saved at:", index_path)

test_index = BASE + "\\outputs\\HBR Case Study_index.pkl"

query = "What is the main challenge discussed in this case?"

db = load_index(test_index)
results = db.similarity_search(query, k=3)

print("🔎 QUERY:", query)
print("\n📌 Top Retrieved Chunks:")

for i, r in enumerate(results):
    print(f"\n----- Chunk {i+1} -----")
    print(r.page_content[:800])

