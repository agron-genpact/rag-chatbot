import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

import streamlit as st

# Choose ONE retriever:
from rag.retriever.vector_retriever import VectorRetriever
# from rag.retriever.hybrid_retriever import HybridRetriever  # hybrid search nice-to-have

from rag.generation.qa import Answerer
from rag.generation.question_rewriter import QuestionRewriter

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot (LangChain Retrieval + OpenAI Generation)")

@st.cache_resource
def load_components():
    # If you want vector-only, return VectorRetriever(), Answerer(), QuestionRewriter()
    return VectorRetriever(), Answerer(), QuestionRewriter()

retriever, answerer, rewriter = load_components()

# -------------------------------
# Memory storage (short-term)
# -------------------------------
if "chat_history" not in st.session_state:
    # Each item: (user_text, assistant_text)
    st.session_state.chat_history = []

if "history" not in st.session_state:
    # For UI rendering / debug
    st.session_state.history = []

question = st.text_input("Ask a question:", placeholder="e.g., What does the document say about ...?")

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Ask") and question.strip():
        user_q = question.strip()

        # -------------------------------
        # 1) Rewrite question using memory (GROUNDING SAFE)
        # -------------------------------
        standalone_q = rewriter.rewrite(st.session_state.chat_history, user_q)

        # -------------------------------
        # 2) Retrieve using rewritten question
        # -------------------------------
        retrieved = retriever.retrieve(standalone_q)

        # -------------------------------
        # 3) Generate answer using ORIGINAL user question + retrieved chunks
        # -------------------------------
        result = answerer.answer(user_q, retrieved)

        # -------------------------------
        # 4) Update memory AFTER answering
        # -------------------------------
        st.session_state.chat_history.append((user_q, result["answer"]))

        # Store for UI view/debug
        st.session_state.history.append((user_q, standalone_q, result, retrieved))

    # Render chat history (latest first)
    for user_q, standalone_q, result, _ in reversed(st.session_state.history):
        st.markdown(f"### Q: {user_q}")
        st.caption(f"Rewritten for retrieval: {standalone_q}")
        st.write(result["answer"])

        if result["citations"]:
            st.markdown("**Citations:**")
            for c in result["citations"]:
                st.markdown(f"- {c}")

with col2:
    st.markdown("### Debug: Retrieved chunks")
    if st.session_state.history:
        _, _, _, retrieved = st.session_state.history[-1]
        for doc, score in retrieved[:5]:
            st.markdown(f"**Score:** {score:.4f}" if score is not None else "**Score:** n/a")
            st.markdown(f"**Source:** {doc.metadata.get('source')}")
            if doc.metadata.get("page") is not None:
                st.markdown(f"**Page:** {doc.metadata.get('page')}")
            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
            st.divider()
