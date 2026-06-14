"""Load settings from .env file."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    namespace: str
    tail_lines: int
    gemini_api_key: str
    gemini_model: str


def load_settings(namespace: str | None = None, tail_lines: int | None = None) -> Settings:
    load_dotenv(BASE_DIR / ".env.example")
    load_dotenv(BASE_DIR / ".env", override=True)
    return Settings(
        namespace=namespace or os.getenv("K8S_NAMESPACE", "default"),
        tail_lines=tail_lines or int(os.getenv("LOG_TAIL_LINES", "200")),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
    )
