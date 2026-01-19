from langchain_text_splitters import RecursiveCharacterTextSplitter

def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Chunking step of ingestion:
    splits text into overlapping chunks.
    Recursive splitter tries to split on paragraphs/sentences first.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )