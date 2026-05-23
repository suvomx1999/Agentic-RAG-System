"""CLI entry point for the evaluator module.

Usage:
    python -m agentic_rag.evaluator --subset all --save
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--subset", default="all", choices=["all", "retrieval", "generation", "e2e"])
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument("--report", type=str, help="Path to results JSON for reporting")
    args = parser.parse_args()

    if args.report:
        from agentic_rag.evaluator.report import main as report_main
        report_main()
        return

    from agentic_rag.evaluator.run_eval import EvalRunner

    runner = EvalRunner()
    runner.load_dataset()
    runner.populate_responses()
    result = runner.evaluate(subset=args.subset)

    if args.save:
        runner.save_results(result)

    # Print summary
    from agentic_rag.evaluator.report import print_summary_table, print_breakdown_table
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        scores = {}
        for col in df.columns:
            if col not in ["user_input", "response", "retrieved_contexts", "reference"]:
                vals = df[col].dropna().tolist()
                if vals:
                    scores[col] = {"mean": sum(vals)/len(vals), "values": vals}
        print_summary_table({"scores": scores})

if __name__ == "__main__":
    main()
