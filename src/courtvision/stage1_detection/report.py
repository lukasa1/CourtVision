"""Sweep reporting — markdown summary + matplotlib plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from courtvision.stage1_detection.sweep import RunSummary


def write_markdown_table(summaries: Sequence[RunSummary], path: Path) -> Path:
    """Write a markdown table of sweep results to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Tag | Knob | Value | n | IDF1 mean | IDF1 std | 95% CI | MOTA mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.tag} | {s.knob or '-'} | {s.value if s.value is not None else '-'} "
            f"| {s.n_clips} | {s.idf1_mean:.4f} | {s.idf1_std:.4f} "
            f"| ±{s.idf1_ci95:.4f} | {s.mota_mean:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def plot_axis(
    summaries: Sequence[RunSummary],
    *,
    axis_name: str,
    baseline_idf1: float | None,
    target_idf1: float,
    output_path: Path,
) -> Path:
    """Plot IDF1 mean ± CI95 as a function of one knob value.

    ``summaries`` should all share the same ``knob`` (the axis being
    swept). Output PNG saved to ``output_path``.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless backend, safe on Modal + CI
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [str(s.value) for s in summaries]
    means = [s.idf1_mean for s in summaries]
    cis = [s.idf1_ci95 for s in summaries]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, means, yerr=cis, fmt="o-", capsize=4, label="dev IDF1 (mean ± 95% CI)")
    if baseline_idf1 is not None:
        ax.axhline(baseline_idf1, color="grey", linestyle="--", label=f"baseline = {baseline_idf1:.3f}")
    ax.axhline(target_idf1, color="red", linestyle=":", label=f"target = {target_idf1:.2f}")
    ax.set_xlabel(axis_name)
    ax.set_ylabel("IDF1")
    ax.set_title(f"Stage 1 sweep: {axis_name}")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


__all__ = ["plot_axis", "write_markdown_table"]
