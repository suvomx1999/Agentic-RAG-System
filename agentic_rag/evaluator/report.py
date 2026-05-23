"""Evaluation report generation with Rich terminal tables.

Provides summary tables, per-type breakdown, and JSON reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from agentic_rag.config import LLM_MODEL, EMBED_MODEL, CHUNK_SIZE, TOP_K, RERANK_TOP_N

console = Console()


def load_results(path: str) -> dict:
    """Load results JSON from disk."""
    with open(path) as f:
        return json.load(f)


def print_summary_table(result: dict):
    """Print a Rich summary table with color-coded scores."""
    table = Table(title="📊 RAGAS Evaluation Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="center")
    table.add_column("Interpretation", justify="center")

    scores = result.get("scores", {})
    for metric, data in scores.items():
        mean = data.get("mean", 0)
        values = data.get("values", [])
        std = (sum((v - mean) ** 2 for v in values) / max(len(values), 1)) ** 0.5 if values else 0

        # Color code
        if mean > 0.8:
            color, interp = "green", "✅ Excellent"
        elif mean > 0.6:
            color, interp = "yellow", "⚠️ Acceptable"
        else:
            color, interp = "red", "❌ Needs Work"

        score_str = f"[{color}]{mean:.3f} ± {std:.3f}[/{color}]"
        table.add_row(metric, score_str, interp)

    console.print(table)


def print_breakdown_table(result: dict):
    """Print per question-type breakdown table."""
    table = Table(title="📋 Breakdown by Question Type", show_lines=True)
    table.add_column("Type", style="bold")
    table.add_column("Count", justify="center")

    by_type = result.get("by_type", {})
    for qtype, count in by_type.items():
        table.add_row(qtype, str(count))

    console.print(table)


def generate_report_json(result: dict) -> dict:
    """Generate structured JSON report with recommendation."""
    scores = result.get("scores", {})
    overall = {m: d["mean"] for m, d in scores.items()}

    weakest = min(overall, key=overall.get) if overall else "N/A"
    strongest = max(overall, key=overall.get) if overall else "N/A"

    report = {
        "timestamp": result.get("timestamp", ""),
        "model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "top_k": TOP_K,
        "rerank_top_n": RERANK_TOP_N,
        "overall_scores": overall,
        "by_type": result.get("by_type", {}),
        "weakest_metric": weakest,
        "strongest_metric": strongest,
        "recommendation": (
            f"The weakest metric is '{weakest}' ({overall.get(weakest, 0):.3f}). "
            f"To improve: consider adjusting retrieval parameters, "
            f"adding more context, or refining the generation prompt."
        ),
    }
    return report


# ── CLI Entry ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="RAGAS evaluation report")
    parser.add_argument("--results", required=True, help="Path or glob to results JSON")
    args = parser.parse_args()

    paths = glob.glob(args.results)
    if not paths:
        print(f"❌ No results found matching: {args.results}")
        return

    for path in paths:
        print(f"\n📄 Loading: {path}")
        result = load_results(path)
        print_summary_table(result)
        print_breakdown_table(result)
        report = generate_report_json(result)
        console.print(f"\n💡 [bold]Recommendation:[/bold] {report['recommendation']}")


if __name__ == "__main__":
    main()
