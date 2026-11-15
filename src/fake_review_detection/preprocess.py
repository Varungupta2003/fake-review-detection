from __future__ import annotations

import re


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
