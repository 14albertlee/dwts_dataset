"""Diagnostic plots for Layer 1 predictions.

    python scripts/plot_layer1.py --mode validate
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwts_model.config import load_config  # noqa: E402

META = {"season", "week", "couple_id", "score"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "holdout"], required=True)
    parser.add_argument("--challenger", default="ridge_phase2a")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    preds = pd.read_csv(cfg.output_dir / f"predictions_{args.mode}.csv")
    models = [c for c in preds.columns if c not in META]
    by_week = pd.read_csv(cfg.output_dir / f"metrics_by_week_{args.mode}.csv")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    for m in models:
        d = by_week[by_week["model"] == m]
        ax.plot(d["week"], d["mae"], marker="o", label=m)
    ax.set(title="MAE by week", xlabel="week", ylabel="MAE")
    ax.legend()

    ax = axes[0, 1]
    for m in models:
        d = by_week[by_week["model"] == m]
        ax.plot(d["week"], d["spearman"], marker="o", label=m)
    ax.set(title="Within-week Spearman by week", xlabel="week", ylabel="Spearman")

    ax = axes[0, 2]
    d = by_week[by_week["model"] == models[0]]
    ax.bar(d["week"], d["n"])
    ax.set(title="Rows per week", xlabel="week", ylabel="n")

    ax = axes[1, 0]
    sc = ax.scatter(preds[args.challenger], preds["score"], c=preds["week"], cmap="viridis", s=14, alpha=0.7)
    ax.plot([3, 10], [3, 10], "k--", lw=1)
    ax.set(title=f"{args.challenger}: predicted vs actual", xlabel="predicted", ylabel="actual")
    fig.colorbar(sc, ax=ax, label="week")

    ax = axes[1, 1]
    for m in models:
        ax.hist(preds[m] - preds["score"], bins=40, alpha=0.4, label=m, histtype="stepfilled")
    ax.axvline(0, color="k", lw=1)
    ax.set(title="Residuals (pred - actual)", xlabel="residual")
    ax.legend()

    ax = axes[1, 2]
    season = preds["season"].max()
    s = preds[preds["season"] == season]
    couples = s.groupby("couple_id")["week"].count().sort_values(ascending=False).index[:4]
    for i, c in enumerate(couples):
        d = s[s["couple_id"] == c].sort_values("week")
        ax.plot(d["week"], d["score"], color=f"C{i}", marker="o", label=f"couple {c} actual")
        ax.plot(d["week"], d[args.challenger], color=f"C{i}", ls="--", marker="x")
    ax.set(title=f"Season {season}: actual (solid) vs {args.challenger} (dashed)", xlabel="week", ylabel="score")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = cfg.output_dir / f"diagnostics_{args.mode}.png"
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")
    print("Bias (mean pred - actual):", preds[models].sub(preds["score"], axis=0).mean().round(3).to_dict())
    print("NaN predictions:", int(preds[models].isna().sum().sum()))


if __name__ == "__main__":
    main()
