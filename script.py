import nbformat as nbf
import os

# Folder where your .py files are located
BASE = r"D:\ADANI CLG\TRI 2\CLASS\GENAI\Assignment\CaseStudySummarizer"

files_to_add = [
    "app.py",
    "rag_pipeline.py",
    "embedder.py",
    "pdf_reader.py",
    "utils.py",
    "preprocess_cases.py"
]

# Output notebook name
output_notebook = "PGDM25_38_Rishil_Assign1.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# Title Cell
title_cell = nbf.v4.new_markdown_cell("""
# PGDM25_38_Rishil_Assign1

**Course:** PGDM AI & DS – Trim 2 – Generative AI  
**Student:** Rishil Shah (Roll No: 38)  
**Assignment:** Case Study Summarizer (RAG-based system using Gemini + FAISS)

This notebook contains the full project code including:
- Frontend (Streamlit App)
- RAG Pipeline
- PDF Reader
- Embedder + FAISS Indexing
- Utility Functions
""")
cells.append(title_cell)

# Add each .py file as a code cell
for filename in files_to_add:
    file_path = os.path.join(BASE, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Markdown header cell
        header = nbf.v4.new_markdown_cell(f"## `{filename}`")
        cells.append(header)

        # Code cell with file content
        code_cell = nbf.v4.new_code_cell(content)
        cells.append(code_cell)
    else:
        print(f"⚠️ File not found: {filename}")

# Add cells to notebook
nb["cells"] = cells

# Save the notebook
output_path = os.path.join(BASE, output_notebook)
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("✅ Notebook created at:", output_path)
