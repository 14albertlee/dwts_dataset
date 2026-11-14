"""Double lift chart: challenger model vs. the best baseline.

Rows are sorted by challenger/baseline prediction ratio and cut into equal-count
bins. Within each bin we plot the mean actual score against both models' mean
predictions. The better model tracks the actual line.

    python scripts/plot_double_lift.py --mode holdout
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

BASELINES = ["last_week", "last_week_drift", "running_mean", "field_adjusted"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "holdout"], required=True)
    parser.add_argument("--challenger", default=None, help="defaults to the main model in config.toml")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    challenger = args.challenger or cfg.main_model
    # The comparison baseline is chosen on validation MAE so holdout is not used to pick it.
    val = pd.read_csv(cfg.output_dir / "metrics_overall_validate.csv")
    val = val[(val["scope"] == "all") & val["model"].isin(BASELINES)]
    baseline = val.loc[val["mae"].idxmin(), "model"]

    preds = pd.read_csv(cfg.output_dir / f"predictions_{args.mode}.csv")
    ch = challenger
    preds["ratio"] = preds[ch] / preds[baseline]
    preds["bin"] = pd.qcut(preds["ratio"].rank(method="first"), args.bins, labels=False)
    g = preds.groupby("bin")[["score", ch, baseline]].mean()

    fig, ax = plt.subplots(figsize=(9, 6))
    x = range(1, args.bins + 1)
    ax.plot(x, g["score"], "k-o", lw=2.5, label="actual")
    ax.plot(x, g[ch], "--s", label=f"{ch} (challenger)")
    ax.plot(x, g[baseline], "--^", label=f"{baseline} (best baseline)")
    ax.set(
        xlabel=f"Bin of {ch} / {baseline} prediction ratio (1 = challenger lowest relative to baseline)",
        ylabel="mean score",
        title=f"Double lift ({args.mode}, n={len(preds)}, {args.bins} equal-count bins)",
    )
    ax.legend()
    fig.tight_layout()
    out = cfg.output_dir / f"double_lift_{args.mode}.png"
    fig.savefig(out, dpi=110)
    print(f"Baseline: {baseline}\nSaved {out}")
    g["ch_err"] = g[ch] - g["score"]
    g["base_err"] = g[baseline] - g["score"]
    print(g.round(3).to_string())
    closer = (g["ch_err"].abs() < g["base_err"].abs()).sum()
    print(f"{ch} closer to actual in {closer}/{args.bins} bins")


if __name__ == "__main__":
    main()
