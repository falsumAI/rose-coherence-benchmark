# MIT License (c) 2025 FalsumAI
"""
A trivial baseline adapter that 'understands' by paraphrasing intent and 'acts' by echoing a canned response.
Useful for smoke tests only.
"""
from __future__ import annotations
from typing import Dict, Any

class EchoAdapter:
    def configure(self, **kwargs): 
        pass

    def infer(self, prompt: str, intent: str) -> Dict[str, str]:
        return {
            "understanding": f"{intent}",
            "action": f"(echo) {prompt.strip()}"
        }
