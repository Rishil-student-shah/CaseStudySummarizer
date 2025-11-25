import pdfplumber
import re

def extract_pdf_text(pdf_path):
    """
    Extract text from PDF with layout=True to capture all lines,
    including quotes and indented 'exclusive' passages that default extraction misses.
    """
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # layout=True is crucial!
            page_text = page.extract_text(layout=True) or ""
            text += page_text + "\n"

    return clean_text(text)



def clean_text(text):
    """
    Clean extracted text: remove page numbers, extra spaces, etc.
    """
    # remove weird unicode
    text = text.encode("ascii", "ignore").decode()

    # remove multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    cleaned = []
    for line in text.split("\n"):
        # remove lines that are only page numbers
        if not re.match(r"^\s*\d+\s*$", line.strip()):
            cleaned.append(line.strip())

    text = "\n".join(cleaned)

    # collapse blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
