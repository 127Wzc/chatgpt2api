from __future__ import annotations

from fastapi import HTTPException

from services.config import config


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(_text(value.get(key)) for key in ("text", "input_text", "content", "input", "instructions", "system", "prompt"))
    return ""


def request_text(*values: object) -> str:
    return "\n".join(part for value in values if (part := _text(value).strip()))


def check_request(text: str) -> None:
    text = str(text or "")
    if not text:
        return
    for word in config.sensitive_words:
        if word in text:
            raise HTTPException(status_code=400, detail={"error": "检测到敏感词，拒绝本次任务"})
