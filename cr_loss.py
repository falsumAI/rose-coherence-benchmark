# MIT License (c) 2025 FalsumAI
"""
Coherence math for Rose Coherence Score (RCS).

We keep it deliberately simple and auditable:
- Convert text to smoothed unigram distributions.
- Compute three directed KL divergences.
- Combine with weights into incoherence energy ℰ.
- Convert to RCS = 1 - min(1, ℰ).
"""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Dict, Tuple

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def _tokens(text: str):
    return _WORD_RE.findall((text or "").lower())

def dist(text: str, vocab: Dict[str, int] | None = None, smoothing: float = 1.0) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Turn text into a probability distribution over tokens with add-one smoothing.
    Returns (probabilities, vocab_counts).
    """
    toks = _tokens(text)
    counts = Counter(toks)
    if vocab is None:
        vocab = dict(counts)
    # Ensure shared support
    for k in list(counts.keys()) + list(vocab.keys()):
        counts.setdefault(k, 0)
        vocab.setdefault(k, 0)
    V = len(vocab) if vocab else 1
    total = sum(counts.values()) + smoothing * V
    probs = {k: (counts.get(k, 0) + smoothing) / total for k in vocab.keys()}
    return probs, counts

def kl(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    """
    KL(p || q) with numeric safety.
    """
    s = 0.0
    for k, pv in p.items():
        qv = q.get(k, eps)
        pv = max(pv, eps)
        qv = max(qv, eps)
        s += pv * math.log(pv / qv)
    return max(0.0, s)

def incoherence_energy(I: str, U: str, A: str, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.5) -> float:
    """
    Compute ℰ = α·KL(I||U) + β·KL(U||A) + γ·KL(A||I)
    We force a shared vocab so divergences are comparable.
    """
    # Build a unified vocab from all channels
    vocab = {}
    for t in (I, U, A):
        for tok in _tokens(t):
            vocab[tok] = vocab.get(tok, 0) + 1
    pI, _ = dist(I, vocab)
    pU, _ = dist(U, vocab)
    pA, _ = dist(A, vocab)
    E = alpha * kl(pI, pU) + beta * kl(pU, pA) + gamma * kl(pA, pI)
    return float(E)

def rcs(I: str, U: str, A: str, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.5) -> float:
    """
    Rose Coherence Score in [0,1].
    """
    E = incoherence_energy(I, U, A, alpha, beta, gamma)
    return float(1.0 - min(1.0, E))

if __name__ == "__main__":
    I = "Sort a list of numbers ascending and explain the algorithm."
    U = "You want ascending sort and an explanation of how it works."
    A = "Here is Python code using sorted(nums) and a short explanation of Timsort."
    print("ℰ =", incoherence_energy(I,U,A))
    print("RCS =", rcs(I,U,A))
