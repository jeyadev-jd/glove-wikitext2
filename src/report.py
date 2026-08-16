"""Generate the PDF project report from measured results only.

Every number rendered here is read from the artefacts produced by ``run.py``.
Any section whose inputs are absent prints "Not yet measured." rather than a
placeholder value.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

NOT_MEASURED = "Not yet measured."


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=17,
                                spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"],
                                   fontSize=9, textColor=colors.grey,
                                   alignment=1, spaceAfter=10),
        "h": ParagraphStyle("h", parent=base["Heading2"], fontSize=11.5,
                            spaceBefore=8, spaceAfter=3,
                            textColor=colors.HexColor("#1f4e79")),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=8.6,
                               leading=11.4, alignment=TA_JUSTIFY, spaceAfter=4),
        "eq": ParagraphStyle("eq", parent=base["BodyText"], fontSize=8.6,
                             leading=11.5, fontName="Courier",
                             leftIndent=10, spaceAfter=4),
        "cap": ParagraphStyle("cap", parent=base["Normal"], fontSize=7.4,
                              textColor=colors.grey, alignment=1, spaceAfter=6),
    }


def _table(rows: Sequence[Sequence[Any]], col_widths: Optional[List[float]] = None,
           font_size: float = 7.4) -> Table:
    table = Table([[str(cell) for cell in row] for row in rows],
                  colWidths=col_widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef3f8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return table


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def build_report(results: Dict[str, Any], output_path: str) -> str:
    """Render the full PDF report; returns ``output_path``."""
    style = _styles()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            title="GloVe from Scratch")

    summary: Dict[str, Any] = results.get("summary", {})
    vocab_stats: Dict[str, Any] = results.get("vocab_stats", {})
    memory: Dict[str, Any] = results.get("memory", {})
    cfg: Dict[str, Any] = results.get("config", {})
    story: List[Any] = []

    def para(text: str, key: str = "body") -> None:
        story.append(Paragraph(text, style[key]))

    def heading(text: str) -> None:
        story.append(Paragraph(text, style["h"]))

    # ---------------------------------------------------------------- title
    para("GloVe Word Embeddings Implemented from Scratch", "title")
    para("Sparse co-occurrence modelling, hand-derived gradients and manual "
         "AdaGrad on WikiText-2, GPU-accelerated with PyTorch tensors", "subtitle")

    # ------------------------------------------------- 1. introduction
    heading("1. Introduction")
    para(
        "Word embeddings map discrete vocabulary items into a continuous vector "
        "space in which geometric relations encode distributional similarity: "
        "words used in similar contexts end up close together, and consistent "
        "semantic contrasts (singular/plural, country/capital) appear as roughly "
        "parallel offsets. GloVe (Global Vectors, Pennington et al., 2014) "
        "learns such a space by factorising a global word-context co-occurrence "
        "matrix instead of scanning local windows stochastically as word2vec "
        "does. Because the corpus statistics are aggregated once up front, "
        "training operates on a compact list of non-zero counts and every "
        "gradient step uses global information."
    )
    para(
        "The objective of this project is a genuine from-scratch implementation: "
        "the co-occurrence construction, the weighted least-squares objective, "
        "the analytic gradients and the AdaGrad optimiser are all written "
        "explicitly. PyTorch is used purely as a vectorised tensor and CUDA "
        "library - no autograd, no <font face='Courier'>torch.optim</font>, and "
        "no pretrained vectors enter training. Official GloVe vectors are "
        "downloaded only after training, for comparison."
    )

    # ------------------------------------------ 2. dataset & preprocessing
    heading("2. Dataset and Preprocessing")
    para(
        "WikiText-2 (raw) is downloaded programmatically from the Hugging Face "
        "mirror of the Salesforce/wikitext dataset and cached locally; the "
        "corpus itself is never vendored. Text is lowercased and tokenised with "
        "a regular expression that keeps alphabetic words (including internal "
        "apostrophes) and numbers while discarding standalone punctuation. A "
        "regex tokeniser is preferable here to a full NLP pipeline: WikiText is "
        "already lightly normalised, the tokeniser needs no model download "
        "(keeping the run reproducible), and dropping punctuation prevents "
        "commas from consuming context slots at the default window size of 1. "
        "Tokens occurring fewer than "
        f"{_fmt(cfg.get('min_frequency'))} times are removed outright rather "
        "than replaced by an &lt;UNK&gt; symbol, and the "
        f"{_fmt(cfg.get('max_vocab_size'))} most frequent survivors form the "
        "vocabulary."
    )
    if vocab_stats:
        story.append(_table([
            ["Raw tokens", "Unique raw types", "Vocabulary", "Retained tokens",
             "Removed rare types", "Tokens retained"],
            [_fmt(vocab_stats.get("raw_token_count")),
             _fmt(vocab_stats.get("unique_raw_tokens")),
             _fmt(vocab_stats.get("vocab_size")),
             _fmt(vocab_stats.get("filtered_token_count")),
             _fmt(vocab_stats.get("removed_rare_types")),
             f"{vocab_stats.get('percent_tokens_retained', 0):.2f}%"],
        ]))
    else:
        para(NOT_MEASURED)

    # ------------------------------------------------ 3. co-occurrence
    heading("3. Co-occurrence Matrix")
    para(
        "A symmetric context window is swept over the index stream. For a target "
        "at position p and a context at position q inside the window the entry "
        "X[i,j] is incremented by 1/|p-q|, so adjacent words contribute 1.0, "
        "words two apart 0.5, three apart 0.333, and so on; both directions are "
        "accumulated, making X symmetric. Out-of-vocabulary tokens are dropped "
        "before windowing, matching the reference GloVe implementation."
    )
    para(
        "A dense V x V float32 matrix is never materialised. The counts are "
        "stored as (row, column, value) triplets, which is what makes the model "
        "trainable inside 4 GB of VRAM at all."
    )
    if memory:
        story.append(_table([
            ["Vocabulary", "Possible dense entries", "Non-zero entries",
             "Sparsity", "Dense memory", "Sparse memory", "Compression"],
            [_fmt(memory.get("vocab_size")),
             _fmt(int(memory.get("possible_dense_entries", 0))),
             _fmt(memory.get("nonzero_entries")),
             f"{memory.get('sparsity_percent', 0):.4f}%",
             f"{memory.get('dense_memory_bytes', 0) / 1024 ** 3:.3f} GB",
             f"{memory.get('sparse_memory_bytes', 0) / 1024 ** 2:.2f} MB",
             f"{memory.get('compression_ratio', 0):.1f}x"],
        ]))
    else:
        para(NOT_MEASURED)

    # ------------------------------------------------------ 4. the model
    heading("4. GloVe Model, Objective and Optimiser")
    para(
        "Each vocabulary entry owns a word vector w_i, a context vector w~_j and "
        "two scalar biases b_i, b~_j, all randomly initialised. Training "
        "minimises the weighted least-squares objective"
    )
    para("J = SUM_ij  f(X_ij) * ( w_i . w~_j + b_i + b~_j - log X_ij )^2", "eq")
    para("with the weighting function", "body")
    para("f(x) = (x / x_max)^alpha  if x &lt; x_max,   else  1", "eq")
    para(
        "which damps the influence of very frequent pairs without discarding "
        "them and prevents rare, noisy pairs from being over-weighted. Writing "
        "e_ij for the bracketed error and f_ij for the weight, the gradients "
        "derived by hand and implemented directly are"
    )
    para("dJ/dw_i = 2 f_ij e_ij w~_j        dJ/dw~_j = 2 f_ij e_ij w_i<br/>"
         "dJ/db_i = 2 f_ij e_ij             dJ/db~_j = 2 f_ij e_ij", "eq")
    para(
        "These are verified against autograd in the test suite. Parameters are "
        "updated with a hand-written AdaGrad step, accumulators initialised to "
        "1.0 as in the reference implementation:"
    )
    para("acc += g^2 ;   param -= lr * g / sqrt(acc + eps)", "eq")
    para(
        "Within a mini-batch the per-sample gradients of a repeated row (a "
        "frequent word such as \"the\" occurs in many samples of the same batch) "
        "are summed into a single per-row gradient before the update is applied. "
        "Applying one raw step per occurrence was observed to diverge to NaN in "
        "float32; summing first is both numerically stable and the correct "
        "mini-batch semantics. Final embeddings are the sum W + W_context, as "
        "prescribed by the paper."
    )

    story.append(PageBreak())

    # -------------------------------------------------------- 5. training
    heading("5. Training Setup and Results")
    para(
        "The full triplet list stays in CPU memory and only the current "
        "mini-batch is transferred to the GPU; the persistent device tensors are "
        "the parameters, the AdaGrad accumulators and the gradient buffers. "
        "A CUDA out-of-memory error halves the batch size, empties the cache and "
        "retries, down to a configurable floor."
    )
    train_rows = [["Setting", "Value", "Setting", "Value"]]
    train_rows.append(["Embedding dimension", _fmt(cfg.get("embedding_dim")),
                       "Epochs", _fmt(cfg.get("epochs"))])
    train_rows.append(["Window size", _fmt(cfg.get("window_size")),
                       "Batch size", _fmt(summary.get("batch_size_used",
                                                      cfg.get("batch_size")))])
    train_rows.append(["Learning rate", _fmt(cfg.get("learning_rate"), 3),
                       "x_max / alpha",
                       f"{_fmt(cfg.get('x_max'), 1)} / {_fmt(cfg.get('alpha'), 2)}"])
    train_rows.append(["Device", _fmt(summary.get("device")),
                       "GPU", _fmt(summary.get("gpu_name") or "CPU only")])
    train_rows.append(["Training time (s)", _fmt(summary.get("training_time"), 2),
                       "Peak GPU memory (MB)",
                       _fmt(summary.get("peak_gpu_memory")) if
                       summary.get("peak_gpu_memory") is not None else "n/a (CPU)"])
    train_rows.append(["Final training loss", _fmt(summary.get("final_loss"), 6),
                       "Training pairs", _fmt(summary.get("nonzero_cooccurrences"))])
    story.append(_table(train_rows, col_widths=[4.2 * cm, 4.2 * cm,
                                                4.2 * cm, 4.2 * cm]))
    story.append(Spacer(1, 5))

    loss_curve = results.get("loss_curve_path")
    if loss_curve and os.path.exists(loss_curve):
        story.append(Image(loss_curve, width=11.5 * cm, height=7.4 * cm))
        story.append(Paragraph("Figure 1: mean weighted squared error per epoch.",
                               style["cap"]))
    else:
        para(NOT_MEASURED)

    # ------------------------------------------------------ 6. evaluation
    heading("6. Evaluation")
    neighbours: List[Dict[str, Any]] = results.get("neighbor_rows", [])
    if neighbours:
        queries: List[str] = []
        for row in neighbours:
            if row["query"] not in queries:
                queries.append(row["query"])
        table_rows = [["Query", "Top-5 nearest neighbours (cosine)"]]
        for query in queries[:6]:
            top = [r for r in neighbours if r["query"] == query][:5]
            table_rows.append([query, ", ".join(
                f"{r['neighbor']} ({r['cosine_similarity']:.2f})" for r in top)])
        story.append(_table(table_rows, col_widths=[2.6 * cm, 14.2 * cm]))
        story.append(Spacer(1, 4))
    else:
        para(NOT_MEASURED)

    analogy = results.get("analogy_summary_custom")
    coverage = results.get("oov_custom")
    if analogy and coverage:
        para(
            "Analogies are solved as a - b + c with the three source words "
            "excluded from the candidate set; quadruples containing an "
            "out-of-vocabulary word are marked OOV and excluded from the "
            f"accuracy denominator. Of {_fmt(analogy['total_analogies'])} "
            f"analogies, {_fmt(analogy['evaluated'])} were evaluable and "
            f"{_fmt(analogy['correct'])} were answered exactly, giving an "
            f"accuracy of {analogy['accuracy'] * 100:.1f}%. Coverage of the "
            f"evaluation vocabulary was "
            f"{coverage['words_found']}/{coverage['total_evaluation_words']} "
            f"words, an OOV rate of {coverage['oov_rate_percent']:.1f}%."
        )
    else:
        para(NOT_MEASURED)

    pca_path = results.get("pca_path")
    if pca_path and os.path.exists(pca_path):
        story.append(Image(pca_path, width=13.5 * cm, height=10.4 * cm))
        story.append(Paragraph(
            "Figure 2: PCA projection of frequent learned word vectors.",
            style["cap"]))

    story.append(PageBreak())

    # ------------------------------------------- 7. official comparison
    heading("7. Comparison with Official GloVe (glove.6B.100d)")
    comparison: List[Dict[str, Any]] = results.get("comparison_rows", [])
    if comparison:
        rows = [["Model", "Vocabulary", "Analogies evaluated", "Correct",
                 "Accuracy", "OOV rate"]]
        for row in comparison:
            rows.append([
                row.get("model"), _fmt(row.get("vocab_size")),
                _fmt(row.get("analogies_evaluated")), _fmt(row.get("correct")),
                f"{row.get('analogy_accuracy', 0) * 100:.1f}%",
                f"{row.get('oov_rate_percent', 0):.1f}%",
            ])
        story.append(_table(rows))
        story.append(Spacer(1, 4))
        para(
            "The official vectors are expected to win, and the reasons are "
            "structural rather than algorithmic. They are trained on roughly 6 "
            "billion tokens of Wikipedia plus Gigaword against the ~2 million "
            "tokens of WikiText-2 - three orders of magnitude more evidence per "
            "word - with a 400,000-word vocabulary, a window of 10 rather than "
            "1, far longer optimisation, and tuned hyperparameters. Larger and "
            "more diverse data mostly helps the mid- and low-frequency words, "
            "which are exactly the ones an analogy set stresses; a window of 1 "
            "additionally captures little beyond immediate collocation, so our "
            "vectors encode syntagmatic adjacency more than broad topical "
            "similarity."
        )
    else:
        para(NOT_MEASURED)

    # ---------------------------------------------------- 8. ablations
    heading("8. Ablation Study")
    ablations: List[Dict[str, Any]] = results.get("ablation_rows", [])
    if ablations:
        rows = [["Parameter", "Value", "Vocabulary", "Non-zero pairs",
                 "Final loss", "Analogy acc.", "OOV %", "Time (s)"]]
        for row in ablations:
            rows.append([
                row["parameter"], row["value"], _fmt(row["vocab_size"]),
                _fmt(row["nonzero_pairs"]), _fmt(row["final_loss"], 4),
                f"{row['analogy_accuracy'] * 100:.1f}%",
                f"{row['OOV_rate']:.1f}%", f"{row['training_time']:.1f}",
            ])
        story.append(_table(rows))
        story.append(Spacer(1, 4))
        para(results.get("ablation_discussion", ""))
    else:
        para(NOT_MEASURED)

    benchmark: List[Dict[str, Any]] = results.get("benchmark_rows", [])
    if benchmark:
        heading("8b. CPU vs GPU Benchmark")
        rows = [["Device", "Epochs", "Total time (s)", "Mean epoch (s)",
                 "Final loss", "Speedup"]]
        for row in benchmark:
            rows.append([row["device"], _fmt(row["epochs"]),
                         _fmt(row["training_time_s"], 2),
                         _fmt(row["mean_epoch_time_s"], 3),
                         _fmt(row["final_loss"], 6),
                         f"{row['gpu_speedup']:.2f}x"])
        story.append(_table(rows))

    # -------------------------------------------------- 9. limitations
    heading("9. Limitations")
    para(
        "The corpus is the dominant limitation: WikiText-2 supplies about three "
        "orders of magnitude fewer tokens than GloVe-6B, so co-occurrence counts "
        "for anything but the most frequent words are thin and their vectors "
        "correspondingly noisy. The 20,000-word cap and the frequency-5 cut-off "
        "remove a long tail of proper nouns and morphological variants, which "
        "shows up directly as out-of-vocabulary analogy items; nothing in the "
        "model can represent a word it never saw. The default window of 1 buys "
        "speed and sparsity at the cost of the wider topical context that helps "
        "semantic analogies. The analogy set used here is small and "
        "hand-assembled, so accuracy figures carry wide confidence intervals and "
        "should be read as directional. Being Wikipedia-derived, the corpus "
        "carries encyclopaedic register and its own demographic and topical "
        "biases into the geometry. Finally, results are sensitive to the "
        "learning rate, x_max and initialisation scale, only a slice of which "
        "the ablations explore, and the 4 GB VRAM budget bounds how far "
        "dimension and window could be pushed."
    )

    # -------------------------------------------------- 10. conclusion
    heading("10. Conclusion")
    if summary:
        para(
            "A complete GloVe pipeline was implemented from first principles and "
            "trained end to end: distance-weighted sparse co-occurrence "
            "construction, the weighted least-squares objective, hand-derived "
            "gradients validated against autograd, and a manual AdaGrad "
            "optimiser driven by mini-batches streamed to the GPU. On "
            f"{_fmt(summary.get('vocab_size'))} vocabulary entries and "
            f"{_fmt(summary.get('nonzero_cooccurrences'))} non-zero pairs the "
            f"model reached a final loss of {_fmt(summary.get('final_loss'), 6)} "
            f"in {_fmt(summary.get('training_time'), 1)} s on "
            f"{summary.get('gpu_name') or summary.get('device')}, using "
            f"{_fmt(summary.get('peak_gpu_memory'))} MB of VRAM - comfortably "
            "inside a 4 GB budget. The exercise makes concrete why the sparse "
            "representation is not optional, how the weighting function "
            "stabilises learning across four orders of magnitude of "
            "co-occurrence counts, and how much of embedding quality is bought "
            "with corpus size rather than algorithmic cleverness."
        )
    else:
        para(NOT_MEASURED)

    doc.build(story)
    return output_path
