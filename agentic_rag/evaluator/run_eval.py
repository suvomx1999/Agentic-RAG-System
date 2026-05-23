"""RAGAS evaluation runner.

Runs queries through LangGraph, collects responses, evaluates with RAGAS.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

CACHE_PATH = Path(__file__).parent / "responses_cache.json"
RESULTS_DIR = Path(__file__).parent


class EvalRunner:
    def __init__(self):
        self._dataset: List[dict] = []
        self._responses: Dict[str, dict] = {}

    def load_dataset(self) -> List[dict]:
        from agentic_rag.evaluator.build_dataset import load_dataset
        self._dataset = load_dataset()
        print(f"📂 Loaded {len(self._dataset)} evaluation samples")
        return self._dataset

    def _load_cache(self) -> Dict[str, dict]:
        if CACHE_PATH.exists():
            with open(CACHE_PATH) as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(CACHE_PATH, "w") as f:
            json.dump(self._responses, f, indent=2)

    def run_graph(self, question: str) -> dict:
        from agentic_rag.agents.graph import build_graph
        from agentic_rag.agents.state import init_state
        graph = build_graph()
        state = init_state(question)
        try:
            final_state = graph.invoke(state)
            answer = final_state.get("answer", "")
            contexts = [d.page_content for d in final_state.get("final_context", [])]
            return {"answer": answer, "contexts": contexts}
        except Exception as e:
            return {"answer": f"Error: {e}", "contexts": []}

    def populate_responses(self):
        self._responses = self._load_cache()
        new_count = 0
        for sample in tqdm(self._dataset, desc="Running queries"):
            question = sample["question"]
            if question in self._responses:
                continue
            result = self.run_graph(question)
            self._responses[question] = result
            new_count += 1
            if new_count % 5 == 0:
                self._save_cache()
        self._save_cache()
        print(f"✅ Populated {new_count} new responses ({len(self._responses)} total)")

    def evaluate(self, subset: str = "all"):
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from agentic_rag.evaluator.metrics import get_metrics
        samples = []
        for entry in self._dataset:
            q = entry["question"]
            resp = self._responses.get(q, {})
            samples.append(SingleTurnSample(
                user_input=q,
                response=resp.get("answer", ""),
                retrieved_contexts=resp.get("contexts", []),
                reference=entry.get("ground_truth", ""),
            ))
        eval_dataset = EvaluationDataset(samples=samples)
        metrics = get_metrics(subset)
        print(f"🔬 Evaluating {len(metrics)} metrics on {len(samples)} samples...")
        return evaluate(dataset=eval_dataset, metrics=metrics)

    def save_results(self, result, suffix: str = "") -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"results_{timestamp}{suffix}.json"
        result_dict = {"timestamp": timestamp, "num_samples": len(self._dataset), "scores": {}}
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            for col in df.columns:
                if col not in ["user_input", "response", "retrieved_contexts", "reference"]:
                    vals = df[col].dropna().tolist()
                    if vals:
                        result_dict["scores"][col] = {"mean": sum(vals)/len(vals), "values": vals}
        type_scores = {}
        for e in self._dataset:
            t = e.get("type", "unknown")
            type_scores[t] = type_scores.get(t, 0) + 1
        result_dict["by_type"] = type_scores
        with open(path, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"💾 Results saved to {path}")
        return path
