from langchain_openai import ChatOpenAI
from rag.generation.prompts import SYSTEM_PROMPT

class QuestionRewriter:
    """
    Uses chat history to rewrite the latest user question into a standalone question.
    """
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    def rewrite(self, chat_history: list[tuple[str, str]], question: str) -> str:
        # Keep it short-term: last few turns
        history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in chat_history[-4:]])

        prompt = f"""Rewrite the user's question into a standalone question.

Chat history:
{history_text}

User question:
{question}

Standalone question:
"""
        resp = self.llm.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt}]
        )
        return resp.content.strip()
