import re

PATTERNS = [
    r"ignore (all|previous) instructions",
    r"system prompt",
    r"developer message",
    r"reveal.*prompt",
    r"bypass",
]

def is_prompt_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in PATTERNS)
