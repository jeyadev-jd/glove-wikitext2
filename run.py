"""End-to-end driver: data -> co-occurrence -> training -> evaluation -> report.

Usage:
    python run.py                     # baseline + evaluation + ablations + report
    python run.py --epochs 2          # quick smoke run
    python run.py --no-ablations      # skip the ablation grid
    python run.py --cpu-benchmark     # add the CPU vs GPU timing comparison
    python run.py --resume            # continue from the latest checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

import pandas as pd
import torch

import config as project_config
from src.data import load_wikitext2_text
from src.evaluation import (evaluate_analogies, evaluate_nearest_neighbors,
                            oov_analysis, select_query_words)
from src.experiments import (CorpusCache, export_vectors, run_ablations,
                             run_cpu_gpu_benchmark, run_pipeline)
from src.pretrained import load_pretrained_index
from src.report import build_report
from src.utils import (print_device_banner, resolve_device, save_json, set_seed)
from src.visualization import (plot_loss_curve, plot_pca_embeddings,
                               select_visualization_words)

RESULTS = project_config.PATHS["results"]


def _write_csv(rows: List[Dict[str, Any]], filename: str) -> str:
    """Write ``rows`` to ``results/<filename>`` and return the path."""
    path = os.path.join(RESULTS, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  wrote {path}")
    return path


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line overrides for the pipeline."""
    parser = argparse.ArgumentParser(description="Train GloVe from scratch.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the configured number of epochs")
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--min-frequency", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--cpu-benchmark", action="store_true")
    parser.add_argument("--benchmark-epochs", type=int, default=3)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="skip the official glove.6B.100d comparison")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the latest checkpoint")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    """Run the whole project pipeline and return the collected results."""
    args = parse_args(argv if argv is not None else sys.argv[1:])
    project_config.ensure_dirs()

    overrides = {
        key: value for key, value in {
            "epochs": args.epochs,
            "embedding_dim": args.embedding_dim,
            "window_size": args.window_size,
            "min_frequency": args.min_frequency,
            "batch_size": args.batch_size,
        }.items() if value is not None
    }
    cfg = project_config.get_config(**overrides)

    print("=" * 72)
    print("GloVe from scratch - WikiText-2")
    print("=" * 72)
    set_seed(int(cfg["seed"]))
    device = resolve_device(args.device)
    device_meta = print_device_banner(device)

    # ------------------------------------------------------------ corpus
    print("\n[1/8] Corpus")
    corpus = CorpusCache(load_wikitext2_text("train"))
    print(f"Tokens after tokenisation: {len(corpus):,}")

    # -------------------------------------------- baseline training run
    print("\n[2/8] Baseline training")
    result = run_pipeline(corpus.tokens, cfg, device, name="custom-wikitext2",
                          checkpoint_dir=project_config.PATHS["checkpoints"],
                          resume=args.resume, verbose=True)
    memory_stats = result.matrix.memory_stats()

    history = result.training["history"].to_dict()
    _write_csv([dict(zip(history, values)) for values in zip(*history.values())],
               "loss_history.csv")
    loss_curve_path = plot_loss_curve(history["epoch"], history["loss"],
                                      os.path.join(RESULTS, "loss_curve.png"))
    print(f"  wrote {loss_curve_path}")

    # -------------------------------------------------------- artefacts
    print("\n[3/8] Exporting vectors")
    exported = export_vectors(result, project_config.PATHS["vectors"])
    for path in exported.values():
        print(f"  wrote {path}")

    # ----------------------------------------------------- evaluation
    print("\n[4/8] Nearest neighbours")
    queries, missing_queries = select_query_words(
        result.index, project_config.NEAREST_NEIGHBOR_QUERIES,
        project_config.NEIGHBOR_FALLBACKS, count=5)
    if missing_queries:
        print(f"  OOV preferred queries replaced: {missing_queries}")
    print(f"  queries used: {queries}")
    neighbor_rows = evaluate_nearest_neighbors(result.index, queries, top_k=10)
    for query in queries:
        top = [r["neighbor"] for r in neighbor_rows if r["query"] == query][:10]
        print(f"  {query:>12}: {', '.join(top)}")

    print("\n[5/8] Analogies")
    analogy_rows, analogy_summary = evaluate_analogies(result.index,
                                                       project_config.ANALOGIES)
    coverage = oov_analysis(result.index, project_config.ANALOGIES, queries)
    print(f"  evaluated {analogy_summary['evaluated']}/"
          f"{analogy_summary['total_analogies']} "
          f"(OOV quadruples: {analogy_summary['oov_analogies']})")
    print(f"  accuracy: {analogy_summary['accuracy'] * 100:.2f}%")
    print(f"  OOV rate over evaluation words: {coverage['oov_rate_percent']:.2f}%")

    # ------------------------------------------------ official comparison
    comparison_rows: List[Dict[str, Any]] = []
    pretrained_index = None
    if not args.no_pretrained:
        print("\n[6/8] Official glove.6B.100d comparison (evaluation only)")
        pretrained_index = load_pretrained_index(verbose=True)
        pre_analogy_rows, pre_summary = evaluate_analogies(pretrained_index,
                                                           project_config.ANALOGIES)
        pre_coverage = oov_analysis(pretrained_index, project_config.ANALOGIES,
                                    queries)
        pre_neighbor_rows = evaluate_nearest_neighbors(pretrained_index, queries,
                                                       top_k=10)
        analogy_rows = analogy_rows + pre_analogy_rows
        neighbor_rows = neighbor_rows + pre_neighbor_rows
        print(f"  official accuracy: {pre_summary['accuracy'] * 100:.2f}% "
              f"(evaluated {pre_summary['evaluated']})")
        for summary_row, cov in ((analogy_summary, coverage),
                                 (pre_summary, pre_coverage)):
            comparison_rows.append({
                "model": summary_row["model"],
                "vocab_size": cov["vocab_size"],
                "analogies_total": summary_row["total_analogies"],
                "analogies_evaluated": summary_row["evaluated"],
                "analogies_oov": summary_row["oov_analogies"],
                "correct": summary_row["correct"],
                "analogy_accuracy": summary_row["accuracy"],
                "evaluation_words": cov["total_evaluation_words"],
                "words_found": cov["words_found"],
                "oov_rate_percent": cov["oov_rate_percent"],
            })
        _write_csv(comparison_rows, "model_comparison.csv")
    else:
        print("\n[6/8] Official GloVe comparison skipped (--no-pretrained)")

    _write_csv(neighbor_rows, "nearest_neighbors.csv")
    _write_csv(analogy_rows, "analogy_results.csv")
    oov_rows = [coverage] + ([pre_coverage] if pretrained_index else [])
    _write_csv(oov_rows, "oov_analysis.csv")

    # ------------------------------------------------------------- PCA
    pca_words = select_visualization_words(
        result.index,
        list(project_config.NEAREST_NEIGHBOR_QUERIES)
        + list(project_config.NEIGHBOR_FALLBACKS)
        + [w for quad in project_config.ANALOGIES for w in quad],
        count=80)
    pca_path = plot_pca_embeddings(result.index, pca_words,
                                   os.path.join(RESULTS, "pca_embeddings.png"),
                                   seed=int(cfg["seed"]))
    print(f"  wrote {pca_path}")

    # ------------------------------------------------------- ablations
    ablation_rows: List[Dict[str, Any]] = []
    run_ablation_flag = project_config.RUN_ABLATIONS and not args.no_ablations
    if run_ablation_flag:
        print("\n[7/8] Ablation experiments")
        ablation_rows = run_ablations(corpus.tokens, cfg, device, verbose=False)
        _write_csv(ablation_rows, "ablation_results.csv")
    else:
        print("\n[7/8] Ablations skipped")

    benchmark_rows: List[Dict[str, Any]] = []
    if args.cpu_benchmark or project_config.RUN_CPU_BENCHMARK:
        print("\n[7b/8] CPU vs GPU benchmark")
        benchmark_rows = run_cpu_gpu_benchmark(corpus.tokens, cfg,
                                               epochs=args.benchmark_epochs)
        _write_csv(benchmark_rows, "cpu_gpu_benchmark.csv")

    # --------------------------------------------------- system metrics
    system_metrics = {
        "device": device_meta,
        "config": cfg,
        "vocabulary": result.vocabulary.stats,
        "cooccurrence": memory_stats,
        "training": {
            "final_loss": result.training["final_loss"],
            "training_time_s": result.training["training_time_s"],
            "batch_size_used": result.training["batch_size_used"],
            "peak_gpu_memory_mb": result.training["peak_gpu_memory_mb"],
            "epochs_completed": len(history["epoch"]),
            "training_pairs": result.training["n_samples"],
        },
        "evaluation": {
            "analogy": analogy_summary,
            "oov": coverage,
            "queries_used": queries,
        },
    }
    save_json(system_metrics, os.path.join(RESULTS, "system_metrics.json"))
    print(f"  wrote {os.path.join(RESULTS, 'system_metrics.json')}")

    summary_row = {
        "vocab_size": len(result.vocabulary),
        "embedding_dimension": cfg["embedding_dim"],
        "window_size": cfg["window_size"],
        "min_frequency": cfg["min_frequency"],
        "nonzero_cooccurrences": result.matrix.nnz,
        "final_loss": result.training["final_loss"],
        "analogy_accuracy": analogy_summary["accuracy"],
        "oov_rate": coverage["oov_rate_percent"],
        "training_time": result.training["training_time_s"],
        "device": str(device),
        "gpu_name": device_meta["gpu_name"] or "",
        "peak_gpu_memory": result.training["peak_gpu_memory_mb"],
        "batch_size_used": result.training["batch_size_used"],
        "epochs": len(history["epoch"]),
    }
    _write_csv([summary_row], "final_summary.csv")

    # ---------------------------------------------------------- report
    print("\n[8/8] Report")
    report_payload = {
        "config": cfg,
        "summary": summary_row,
        "vocab_stats": result.vocabulary.stats,
        "memory": memory_stats,
        "loss_curve_path": loss_curve_path,
        "pca_path": pca_path,
        "neighbor_rows": [r for r in neighbor_rows
                          if r["model"] == "custom-wikitext2"],
        "analogy_summary_custom": analogy_summary,
        "oov_custom": coverage,
        "comparison_rows": comparison_rows,
        "ablation_rows": ablation_rows,
        "ablation_discussion": _ablation_discussion(ablation_rows),
        "benchmark_rows": benchmark_rows,
    }
    report_path = build_report(report_payload,
                               os.path.join(project_config.PATHS["report"],
                                            "glove_report.pdf"))
    print(f"  wrote {report_path}")

    print("\nDone.")
    return report_payload


def _ablation_discussion(rows: List[Dict[str, Any]]) -> str:
    """Compose a factual paragraph describing the measured ablation outcomes."""
    if not rows:
        return ""
    parts: List[str] = []
    for parameter in sorted({row["parameter"] for row in rows}):
        subset = [row for row in rows if row["parameter"] == parameter]
        best_loss = min(subset, key=lambda r: r["final_loss"])
        best_acc = max(subset, key=lambda r: r["analogy_accuracy"])
        parts.append(
            f"For {parameter}, the lowest final loss "
            f"({best_loss['final_loss']:.4f}) came from {parameter}="
            f"{best_loss['value']}, while the best analogy accuracy "
            f"({best_acc['analogy_accuracy'] * 100:.1f}%) came from {parameter}="
            f"{best_acc['value']}"
        )
    overall = max(rows, key=lambda r: r["analogy_accuracy"])
    parts.append(
        f"Across the whole grid the strongest analogy accuracy was "
        f"{overall['analogy_accuracy'] * 100:.1f}% at {overall['parameter']}="
        f"{overall['value']}. Note that final loss and analogy accuracy are not "
        "interchangeable: raising the embedding dimension or the epoch count "
        "lowers the training objective almost by construction, which says "
        "nothing on its own about how well the geometry generalises"
    )
    return ". ".join(parts) + "."


if __name__ == "__main__":
    main()
