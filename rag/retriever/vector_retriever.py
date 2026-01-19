import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'vectorestore')

TOP_K = 5

class VectorRetriever:
    """
    Retrieval step (runtime):
    - load FAISS index
    - retrieve Top-K chunks for a query
    """
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
        self.vs = FAISS.load_local(INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True)

    def retrieve(self, query: str, k: int = TOP_K):
        # similarity_search_with_score returns (Document, score) pairs
        results = self.vs.similarity_search_with_score(query, k=k)

        print("Top retrieved results:")
        for doc, score in results[:3]:
            print(f"- score={score:.4f} | source={doc.metadata.get('source')} page={doc.metadata.get('page')}")

        return results
