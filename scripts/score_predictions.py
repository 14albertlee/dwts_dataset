"""Re-score saved predictions without refitting any model.

    python scripts/score_predictions.py --mode validate
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwts_model.config import load_config  # noqa: E402
from dwts_model.evaluate import summarize  # noqa: E402

META = {"season", "week", "couple_id", "score"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "holdout"], required=True)
    parser.add_argument("--confirm-holdout", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.mode == "holdout" and not args.confirm_holdout:
        sys.exit("Pass --confirm-holdout to score holdout predictions.")

    cfg = load_config(args.config)
    preds = pd.read_csv(cfg.output_dir / f"predictions_{args.mode}.csv")
    models = [c for c in preds.columns if c not in META]
    tables = summarize(preds, models, cfg.short_seasons)
    for key, table in tables.items():
        table.to_csv(cfg.output_dir / f"metrics_{key}_{args.mode}.csv", index=False)

    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 160
    overall = tables["overall"]
    print(overall[overall["scope"] == "all"].drop(columns="scope").to_string(index=False))


if __name__ == "__main__":
    main()
