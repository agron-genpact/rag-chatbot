import re

def clean_text(text: str) -> str:
    """
    Cleaning step of ingestion:
    - normalize whitespace
    - remove weird non-breaking spaces
    """
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()