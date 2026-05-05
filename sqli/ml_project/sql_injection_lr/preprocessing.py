"""SQL-aware text cleaning and tokenization."""

from __future__ import annotations

import html
import re

SQL_RELEVANT_CHARS = set("'\"=<>-;*/(),#@+%._")

TOKEN_RE = re.compile(
    r"--|/\*|\*/|!=|<>|<=|>=|=|"
    r"[a-z_][a-z0-9_]*|"
    r"\d+(?:\.\d+)?|"
    r"'|\"|;|\*|,|\(|\)|<|>|#|@|\+|%|-|/|\."
)


def clean_text(text: str) -> str:
    """Lowercase, decode HTML entities, and keep SQL-relevant symbols."""
    decoded = html.unescape(text or "").lower()
    kept: list[str] = []
    for char in decoded:
        if char.isalnum() or char.isspace() or char in SQL_RELEVANT_CHARS:
            kept.append(char)
        else:
            kept.append(" ")
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize without losing SQL operators and comment markers."""
    cleaned = clean_text(text)
    return TOKEN_RE.findall(cleaned)


def make_ngrams(tokens: list[str], ngram_min: int = 1, ngram_max: int = 2) -> list[str]:
    terms: list[str] = []
    upper = max(ngram_min, ngram_max)
    for n in range(max(1, ngram_min), upper + 1):
        if len(tokens) < n:
            continue
        if n == 1:
            terms.extend(tokens)
        else:
            terms.extend(" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    return terms
