import os

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'vectorestore')

TOP_K = 5


class HybridRetriever:
    """
    Hybrid = BM25 (keywords) + FAISS (vectors)
    Simple approach:
    - retrieve top_k from both
    - merge + de-duplicate (you can also score-fuse)
    """
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vs = FAISS.load_local(INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True)

        # Build BM25 from FAISS docstore contents
        # FAISS store has a docstore with original documents
        all_docs = list(self.vs.docstore._dict.values())
        self.bm25 = BM25Retriever.from_documents(all_docs)
        self.bm25.k = TOP_K

    def retrieve(self, query: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
        vec_results = self.vs.similarity_search_with_score(query, k=k)
        bm25_docs = self.bm25.invoke(query)

        # Merge (keep doc uniqueness by content+meta)
        seen = set()
        merged: List[Tuple[Document, float]] = []

        for doc, score in vec_results:
            key = (doc.page_content[:200], tuple(sorted(doc.metadata.items())))
            if key not in seen:
                merged.append((doc, score))
                seen.add(key)

        # BM25 has no cosine score; give a placeholder high score so it is kept
        for doc in bm25_docs:
            key = (doc.page_content[:200], tuple(sorted(doc.metadata.items())))
            if key not in seen:
                merged.append((doc, 1.0))
                seen.add(key)

        print(f"Hybrid retrieved total={len(merged)} (vec={len(vec_results)}, bm25={len(bm25_docs)})")
        return merged[:k]
