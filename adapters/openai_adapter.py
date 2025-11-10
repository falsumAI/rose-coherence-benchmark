# MIT License (c) 2025 FalsumAI
"""
Minimal OpenAI adapter example. Requires 'openai' package and an API key.
User must install extra dep: pip install openai
"""
from __future__ import annotations
from typing import Dict, Any
import os

class OpenAIAdapter:
    def __init__(self):
        self.api_key = None
        self.model = None

    def configure(self, openai_api_key=None, openai_model=None, **_):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise RuntimeError("OpenAI API key required. Pass --openai_api_key or set OPENAI_API_KEY.")

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError("Install the OpenAI SDK: pip install openai") from e

    def infer(self, prompt: str, intent: str) -> Dict[str, str]:
        # Two-turn protocol: first get U (paraphrase), then A (solution)
        u_msg = f"Paraphrase the user's intent in one sentence so a junior engineer would not misinterpret it.\nIntent: {intent}"
        a_msg = f"You must now act on this intent. Task: {prompt}\nRespond with the final action/output only."
        U = self._chat(u_msg)
        A = self._chat(a_msg)
        return {"understanding": U.strip(), "action": A.strip()}

    def _chat(self, content: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":content}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
