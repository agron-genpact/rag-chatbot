from langchain_core.documents import Document

def citation_for(doc: Document) -> str:
    """
    Turn metadata into a human-readable citation.
    """
    src = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page")
    if page is not None:
        return f"{src} (page {page})"
    return f"{src}"