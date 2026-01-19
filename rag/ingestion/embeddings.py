import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../..', '.env'))

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# from rag.config import DATA_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP
# from rag.logging_utils import get_logger
from loaders import load_all
from cleaners import clean_text
from chunks import make_splitter

# log = get_logger("build_index")

DATA_DIR = os.path.join(os.path.dirname(__file__), '../..', 'data')
INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'vectorestore')

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def main():
    # 1) LOAD
    raw_docs = load_all(DATA_DIR)
    # log.info(f"Loaded raw docs/pages: {len(raw_docs)} from {DATA_DIR}")

    # Convert to LangChain Document objects + metadata
    lc_docs: list[Document] = []
    for d in raw_docs:
        text = clean_text(d.text)  # 2) CLEAN
        if not text:
            continue

        meta = {"source": d.source}
        if d.page is not None:
            meta["page"] = d.page

        lc_docs.append(Document(page_content=text, metadata=meta))

    # log.info(f"After cleaning: {len(lc_docs)} docs")

    # 3) CHUNK
    splitter = make_splitter(CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = splitter.split_documents(lc_docs)
    # log.info(f"Created chunks: {len(chunks)}")
    # log.info(f"Sample chunk meta: {chunks[0].metadata if chunks else 'n/a'}")

    print(f"Created chunks: {len(chunks)}")
    # 4) EMBED + 5) INDEX
    # Embeddings model that calls OpenAI embeddings API.
    # (You can swap to local embeddings later if you want.)
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

    # Create FAISS vector store from chunks
    vs = FAISS.from_documents(chunks, embeddings)  # from_documents is a standard FAISS init :contentReference[oaicite:5]{index=5}

    # Save index to disk
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    # log.info(f"Saved FAISS index to {INDEX_DIR}")

    # # Debug print: show first chunk preview
    # if chunks:
    #     log.info(f"First chunk preview:\n{chunks[0].page_content[:250]}...")

if __name__ == "__main__":
    main()
