# MIT License (c) 2025 FalsumAI
from __future__ import annotations
import argparse, json, time, pathlib, importlib
from typing import Dict, Any
from tqdm import tqdm
from cr_loss import rcs

def load_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def choose_adapter(name: str):
    if name == "echo":
        from adapters.echo import EchoAdapter
        return EchoAdapter()
    elif name == "openai":
        from adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter()
    else:
        # dynamic import for user-defined adapters
        spec = importlib.import_module(f"adapters.{name}")
        return spec.Adapter()

def run(tasks_path: str, model_name: str, **kwargs):
    tasks = load_tasks(tasks_path)
    adapter = choose_adapter(model_name)
    adapter.configure(**kwargs)

    scores = []
    rows = []
    for t in tqdm(tasks, desc="RCS"):
        intent = t["intent"]
        prompt = t["prompt"]
        ref = t.get("reference_action","")
        out = adapter.infer(prompt=prompt, intent=intent)
        U = out.get("understanding","")
        A = out.get("action","")
        score = rcs(intent, U, A, alpha=t.get("alpha",1.0), beta=t.get("beta",0.5), gamma=t.get("gamma",0.5))
        rows.append({
            "task_id": t.get("id"),
            "rcs": round(score,4),
            "intent": intent,
            "understanding": U,
            "action": A,
            "reference_action": ref
        })
        scores.append(score)

    avg = sum(scores)/len(scores) if scores else 0.0
    results = {"model": model_name, "average_rcs": round(avg,4), "results": rows}
    outpath = pathlib.Path("results")
    outpath.mkdir(exist_ok=True, parents=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    fp = outpath / f"rcs_{model_name}_{stamp}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAverage RCS: {avg:.4f}\nSaved: {fp}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="tasks/sample_tasks.json")
    p.add_argument("--model", default="echo")
    # passthroughs (e.g., API keys) are captured and forwarded
    p.add_argument("--openai_api_key", default=None)
    p.add_argument("--openai_model", default=None)
    args = p.parse_args()
    out = run(
    tasks_path=args.tasks,
    model_name=args.model,
    openai_api_key=args.openai_api_key,
    openai_model=args.openai_model
)

# Pretty RCS score summary
scores = [item.get("rcs") for item in out.get("results", []) if "rcs" in item]
if scores:
    avg = sum(scores) / len(scores)
    print("\n==============================")
    print(f" Model: {args.openai_model or args.model}")
    print(f" RCS Score: {avg:.3f}")
    print("==============================\n")
else:
    print("No RCS scores found in results.")

