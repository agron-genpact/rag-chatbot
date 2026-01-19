from langchain_openai import ChatOpenAI
# from rag.config import OPENAI_MODEL, MIN_SCORE
# from rag.logging_utils import get_logger
from rag.generation.prompts import SYSTEM_PROMPT
from rag.utils.citations import citation_for
from rag.generation.guardrails import is_prompt_injection

OPENAI_MODEL = "gpt-4o-mini" 
MIN_SCORE = 0.30


# log = get_logger("answerer")




class Answerer:
    """
    Generation step (runtime):
    - take retrieved chunks
    - build a grounded prompt
    - call OpenAI chat model
    - return answer + citations
    """
    def __init__(self):
        self.llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)  # ChatOpenAI is LangChain OpenAI chat wrapper :contentReference[oaicite:7]{index=7}

    def _build_context(self, retrieved):
        # retrieved is list[(Document, score)]
        context_blocks = []
        citations = []

        for doc, score in retrieved:
            # IMPORTANT: score meaning can vary by store; treat this as a relevance signal.
            # We use MIN_SCORE as a conservative cutoff.
            if score is not None and score < MIN_SCORE:
                continue

            cite = citation_for(doc)
            citations.append(cite)

            # Put citation label adjacent to the chunk so the model can cite correctly
            context_blocks.append(f"[{cite}]\n{doc.page_content}")

        return context_blocks, citations

    def answer(self, question: str, retrieved):
        if is_prompt_injection(question):
            return {
                "answer": "I can’t help with requests to override instructions. Ask a question about the provided documents.",
                "citations": [],
            }
        
        context_blocks, citations = self._build_context(retrieved)

        if not context_blocks:
            return {
                "answer": "I don’t know based on the provided documents.",
                "citations": [],
            }

        user_prompt = f"""Question:
        {question}

        Context (use ONLY this):
        {chr(10).join(context_blocks)}

        Write an answer grounded in the context and include citations like [source (page)].
        """

        print(f"Sending {len(context_blocks)} chunks to LLM")
        resp = self.llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        return {
            "answer": resp.content,
            "citations": list(dict.fromkeys(citations)),  # unique citations, preserve order
        }
