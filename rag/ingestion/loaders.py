import os
from typing import List
from dataclasses import dataclass

from pypdf import PdfReader
from docx import Document as DocxDocument

@dataclass
class LoadedDoc:
    """
    A minimal document container before we convert to LangChain Documents.
    """
    text: str
    source: str          # file name
    page: int | None     # PDF page number if available

def load_pdf(path: str) -> List[LoadedDoc]:
    reader = PdfReader(path)
    out: List[LoadedDoc] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        out.append(LoadedDoc(text=text, source=os.path.basename(path), page=i + 1))
    return out

def load_docx(path: str) -> List[LoadedDoc]:
    doc = DocxDocument(path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [LoadedDoc(text=text, source=os.path.basename(path), page=None)]

def load_md(path: str) -> List[LoadedDoc]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [LoadedDoc(text=f.read(), source=os.path.basename(path), page=None)]

def load_all(data_dir: str) -> List[LoadedDoc]:
    """
    Load every PDF/MD/DOCX from data_dir recursively.
    """
    docs: List[LoadedDoc] = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            p = os.path.join(root, fn)
            low = fn.lower()
            if low.endswith(".pdf"):
                docs.extend(load_pdf(p))
            elif low.endswith(".docx"):
                docs.extend(load_docx(p))
            elif low.endswith(".md"):
                docs.extend(load_md(p))
    return docs
