## Results (live)

- Latest run: **echo adapter**  
- **Average RCS:** `0.7728`  
- JSON: [`rcs_echo_20251109-230939.json`](results/rcs_echo_20251109-230939.json)




> RCS ranges 0.00–1.00 (higher = more coherent). Echo is a loopback sanity check; model runs (gpt-4o-mini, etc.) will be added as quota is enabled.
# Rose Coherence Benchmark (RCS)

**Scale:** 0.00 → 1.00 (higher is more coherent)  
**Thesis:** Formatting is superficial. **Coherence**—alignment between *Intent (I)*, *Understanding (U)*, and *Action (A)*—is what matters.

We operationalize incoherence energy:

\[
\mathcal{E} = \alpha\,KL(I\parallel U) + \beta\,KL(U\parallel A) + \gamma\,KL(A\parallel I)
\]

and define the **Rose Coherence Score**:

\[
\mathrm{RCS} = 1 - \min(1, \mathcal{E})
\]

This repo provides a tiny, model-agnostic harness so anyone can score any model on the same truth-preserving reasoning protocol.

---

## Quick start

```bash
# 1) Create a venv (Python 3.9+)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps (none are heavyweight)
pip install -r requirements.txt

# 3) Run the benchmark with the built-in Echo model (for smoke test)
python benchmark.py --model echo --tasks tasks/sample_tasks.json

# 4) Plug your own model by implementing adapters/your_model.py (see openai_adapter.py as a template)
python benchmark.py --model openai --tasks tasks/sample_tasks.json \
  --openai_api_key $OPENAI_API_KEY --openai_model gpt-4o-mini  # example
```

The harness expects each model to return **U** (*understanding* paraphrase) and **A** (*final action/output*). Adapters can produce U by prompting the model to restate the intent before answering, or infer U from tool traces if available.

---

## Files

- `cr_loss.py` – Coherence math: token-distribution with add-one smoothing → `KL(p||q)`, `incoherence_energy`, `rcs` (1-ℰ clipped).  
- `benchmark.py` – Loads tasks, queries a model adapter, computes per-task and aggregate RCS.  
- `adapters/` – Minimal interface; includes `echo` (baseline) and `openai_adapter` (example).  
- `tasks/sample_tasks.json` – Small, realistic tasks with ground-truth intents and references.  
- `requirements.txt` – `tqdm`, `numpy` only. Keep it light.  
- `LICENSE` – MIT.

---

## Method (simple, auditable)

1) Convert each text channel (I, U, A) into a **smoothed token distribution** over unigrams (lowercased, alnum).  
2) Compute three directed **KL divergences**: `KL(I||U)`, `KL(U||A)`, `KL(A||I)`.  
3) Weighted sum → **incoherence energy** `ℰ`. Defaults: `α=1.0, β=0.5, γ=0.5`.  
4) Report **RCS = 1 - min(1, ℰ)** and macro-average across tasks.

This is intentionally minimal and deterministic. You can swap a richer encoder (e.g., embeddings) as long as you keep the I/U/A contract.

---

## Why this matters

Benchmarks that reward *style* produce shallow gains. RCS rewards **reasoning integrity**: did the system understand the user and act accordingly, while staying true to the stated intent?

Use it to compare models, tune prompts, track regressions, or as a safety gate (reject low RCS).

---

## Citation

> Rose Coherence Benchmark (RCS). FalsumAI — 2025. MIT License.
