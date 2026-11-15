"""OLS effects report: how much each Ridge predictor moves the average judge score.

    python scripts/report_ols.py            # -> outputs/reports/ols/report.md (+ CSVs and PNGs)

Fits OLS on every season with the exact feature sets of the pre-show (`[model].main`) and
order-announced (`[model].order`) Ridge models. The target is the couple's weekly average judge score
(per judge, per dance; 1-10). Standard errors are clustered by couple, since each couple contributes
one row per week. Descriptive only: the forecasts still come from the Ridge models.
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwts_model.config import load_config  # noqa: E402
from dwts_model.data import load_performance_context, load_weekly_scores  # noqa: E402
from dwts_model.feature_engineering import build_context_features, build_features  # noqa: E402
from dwts_model.models import RidgeModel  # noqa: E402
from dwts_model.reporting import md_table  # noqa: E402

OTHER = "style_other"  # dances outside the one-hot list: the omitted style
READINGS = {
    "prev_score": "+1 point in last week's score (running mean held fixed)",
    "running_mean": "+1 point in the average of all earlier weeks (last week held fixed)",
    "form": "+1 point in every earlier week (last week and running mean both +1)",
    "week": "one week later in the season",
    "theme_personal_emotional": "a personal / emotional theme week",
    "prev_bottom3": "in the bottom 3 of judge scores last week",
    "order_frac": "dancing last in the round instead of first",
}
GROUPS = {
    "form (prev_score, running_mean, prev_bottom3)": ["prev_score", "running_mean", "prev_bottom3"],
    "week": ["week"],
    "theme": ["theme_personal_emotional"],
    "dance style": None,  # every style_* column
    "running order": ["order_frac"],
}


def style_name(col: str) -> str:
    return col.removeprefix("style_").replace("_", " ").title().replace("Cha Cha Cha", "Cha-cha-cha")


def fit_ols(df: pd.DataFrame, feats: list[str], cov: str = "cluster"):
    X = sm.add_constant(df[feats].astype(float))
    model = sm.OLS(df["score"].astype(float), X)
    if cov == "cluster":
        return model.fit(cov_type="cluster", cov_kwds={"groups": df["cluster"].to_numpy()})
    return model.fit(cov_type=cov)


def contrast(res, weights: dict[str, float]) -> tuple[float, float]:
    """Estimate and standard error of sum(w * coef)."""
    names = list(res.params.index)
    c = np.zeros(len(names))
    for k, w in weights.items():
        c[names.index(k)] = w
    est = float(c @ res.params.to_numpy())
    se = float(np.sqrt(c @ res.cov_params().to_numpy() @ c))
    return est, se


def effects_table(res, df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Plain-unit effects: non-style features per unit (or 0->1), styles as deviation from the average dance."""
    z = 1.959964
    styles = [f for f in feats if f.startswith("style_")]
    rows = []
    for f in [f for f in feats if not f.startswith("style_")]:
        x = df[f].astype(float)
        binary = set(x.dropna().unique()) <= {0.0, 1.0} or f == "order_frac"
        est, se = res.params[f], res.bse[f]
        swing_lo, swing_hi = (0.0, 1.0) if binary else tuple(x.quantile([0.1, 0.9]))
        rows.append({
            "kind": "predictor", "term": f, "reading": READINGS.get(f, f), "effect": est, "lo": est - z * se,
            "hi": est + z * se, "p": res.pvalues[f], "sd_x": x.std(), "std_effect": est * x.std(),
            "swing": est * (swing_hi - swing_lo), "swing_range": f"{swing_lo:.3g} to {swing_hi:.3g}",
        })
    if {"prev_score", "running_mean"} <= set(feats):
        est, se = contrast(res, {"prev_score": 1.0, "running_mean": 1.0})
        rows.append({"kind": "combined", "term": "form", "reading": READINGS["form"], "effect": est,
                     "lo": est - z * se, "hi": est + z * se, "p": 2 * (1 - _norm_cdf(abs(est / se)))})
    if styles:
        # Share of all performances per style; the omitted "other" styles take the remainder.
        w = df[styles].astype(float).mean()
        w[OTHER] = max(0.0, 1.0 - w.sum())
        for s in [*styles, OTHER]:
            weights = {k: -w[k] for k in styles}
            if s != OTHER:
                weights[s] += 1.0
            est, se = contrast(res, weights)
            rows.append({"kind": "style", "term": s, "reading": f"{style_name(s)} vs the average dance",
                         "effect": est, "lo": est - z * se, "hi": est + z * se,
                         "p": 2 * (1 - _norm_cdf(abs(est / se))), "share_of_dances": w[s]})
    return pd.DataFrame(rows)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def group_importance(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Drop-group loss in R^2 and Shapley (LMG) share of R^2 for each predictor group."""
    groups = {}
    for name, cols in GROUPS.items():
        cols = [f for f in feats if f.startswith("style_")] if cols is None else [c for c in cols if c in feats]
        if cols:
            groups[name] = cols
    names = list(groups)
    y = df["score"].astype(float)
    r2 = {(): 0.0}
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(names, k):
            cols = [c for g in combo for c in groups[g]]
            r2[combo] = sm.OLS(y, sm.add_constant(df[cols].astype(float))).fit().rsquared
    full = r2[tuple(names)]
    G = len(names)
    rows = []
    for g in names:
        others = [n for n in names if n != g]
        shap = 0.0
        for k in range(len(others) + 1):
            for combo in itertools.combinations(others, k):
                with_g = tuple(n for n in names if n in combo or n == g)
                weight = math.factorial(k) * math.factorial(G - k - 1) / math.factorial(G)
                shap += weight * (r2[with_g] - r2[tuple(combo)])
        rows.append({"group": g, "n_terms": len(groups[g]), "r2_alone": r2[(g,)],
                     "r2_drop": full - r2[tuple(others)], "shapley_r2": shap, "share_of_r2": shap / full})
    return pd.DataFrame(rows).sort_values("shapley_r2", ascending=False)


def vif_table(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = sm.add_constant(df[feats].astype(float)).to_numpy()
    return pd.DataFrame({"term": feats, "vif": [variance_inflation_factor(X, i + 1) for i in range(len(feats))]})


def ridge_vs_ols(df: pd.DataFrame, feats: list[str], alpha: float, res) -> pd.DataFrame:
    """Raw-unit coefficients of both fits; styles vs the average dance (the omitted-style baseline differs between fits)."""
    ridge = RidgeModel(feats, alpha).fit(df)
    pipe = ridge.pipe_
    coef = pipe.named_steps["ridge"].coef_[: len(feats)] / pipe.named_steps["standardscaler"].scale_[: len(feats)]
    out = pd.DataFrame({"term": feats, "ols": res.params[feats].to_numpy(), "ridge": coef})
    styles = [f for f in feats if f.startswith("style_")]
    if styles:
        w = df[styles].astype(float).mean()
        is_style = out["term"].isin(styles)
        for col in ("ols", "ridge"):
            out.loc[is_style, col] -= (out.loc[is_style, col].to_numpy() * w[styles].to_numpy()).sum()
        out.loc[is_style, "term"] += " (vs average dance)"
    if {"prev_score", "running_mean"} <= set(feats):
        form = out[out["term"].isin(["prev_score", "running_mean"])][["ols", "ridge"]].sum()
        out = pd.concat([out, pd.DataFrame([{"term": "form (prev_score + running_mean)", **form}])], ignore_index=True)
    out["ridge_minus_ols"] = out["ridge"] - out["ols"]
    return out


def robustness(df: pd.DataFrame, feats: list[str], split: int) -> pd.DataFrame:
    early, late = df[df["season"] <= split], df[df["season"] > split]
    fe = pd.get_dummies(df["season"], prefix="season", drop_first=True, dtype=float)
    fits = {
        "all seasons": fit_ols(df, feats).params,
        f"seasons 1-{split}": fit_ols(early, feats).params,
        f"seasons {split + 1}+": fit_ols(late, feats).params,
        "season fixed effects": fit_ols(pd.concat([df, fe], axis=1), [*feats, *fe.columns]).params,
    }
    return pd.DataFrame({k: v[feats] for k, v in fits.items()}).rename_axis("term").reset_index()


def forest(ax, eff: pd.DataFrame, title: str) -> None:
    e = eff[eff["kind"] != "combined"].copy()
    e["label"] = [r.term if r.kind == "predictor" else style_name(r.term) for r in e.itertuples()]
    e = pd.concat([e[e["kind"] == "predictor"], e[e["kind"] == "style"].sort_values("effect")])
    y = np.arange(len(e))[::-1]
    colors = ["tab:blue" if k == "predictor" else "tab:orange" for k in e["kind"]]
    ax.errorbar(e["effect"], y, xerr=[e["effect"] - e["lo"], e["hi"] - e["effect"]], fmt="none", ecolor=colors, lw=1.5)
    ax.scatter(e["effect"], y, c=colors, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, e["label"])
    ax.set(xlabel="change in average judge score (points)", title=title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--split", type=int, default=16, help="last season of the early era in the robustness check")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.output_dir.parent / "reports" / "ols"
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = json.loads((cfg.output_dir / "selected_params.json").read_text())["ridge_alpha"]
    history = build_features(load_weekly_scores(cfg), build_context_features(load_performance_context(cfg)))
    history["cluster"] = history["season"] * 10000 + history["couple_id"]

    models = {"pre-show": cfg.main_model, "order-announced": cfg.order_model}
    results, sections = {}, []
    for label, model_name in models.items():
        set_name = model_name.removeprefix("ridge_")
        feats = list(cfg.features[set_name])
        df = history.dropna(subset=feats + ["score"]).reset_index(drop=True)
        dropped = len(history) - len(df)
        res = fit_ols(df, feats)
        hc1 = fit_ols(df, feats, cov="HC1")
        eff = effects_table(res, df, feats)
        imp = group_importance(df, feats)
        vif = vif_table(df, feats)
        rvo = ridge_vs_ols(df, feats, alphas[set_name], res)
        rob = robustness(df, feats, args.split)
        se = pd.DataFrame({"term": feats, "se_cluster": res.bse[feats].to_numpy(), "se_hc1": hc1.bse[feats].to_numpy()})
        slug = set_name
        for name, t in (("effects", eff), ("importance", imp), ("vif", vif), ("ridge_vs_ols", rvo),
                        ("robustness", rob), ("se_check", se)):
            t.to_csv(out_dir / f"{name}_{slug}.csv", index=False)
        results[label] = dict(model=model_name, res=res, eff=eff, imp=imp, df=df, feats=feats, dropped=dropped)

        pred = eff[eff["kind"].isin(["predictor", "combined"])].copy()
        pred["swing_range"] = pred["swing_range"].fillna("")
        sty = eff[eff["kind"] == "style"].sort_values("effect", ascending=False)
        top, bottom = sty.iloc[0], sty.iloc[-1]
        imp_top = imp.iloc[0]
        sentences = []
        for r in pred.itertuples():
            sig = "" if r.p < 0.05 else " (not distinguishable from zero at 95%)"
            sentences.append(f"{r.reading}: {r.effect:+.2f} points [{r.lo:+.2f}, {r.hi:+.2f}]{sig}.")
        sentences.append(
            f"Dance style: {style_name(top.term)} scores highest ({top.effect:+.2f} vs the average dance) and "
            f"{style_name(bottom.term)} lowest ({bottom.effect:+.2f}), a spread of {top.effect - bottom.effect:.2f} points."
        )
        sentences.append(
            f"{imp_top.group} explains most of what the model explains ({imp_top.share_of_r2:.0%} of R^2 = {res.rsquared:.3f})."
        )
        high_vif = vif[vif["vif"] > 5]
        sections += [
            f"## {label.capitalize()} model (`{model_name}`)",
            "",
            f"OLS on {len(df):,} couple-weeks, seasons {int(df['season'].min())}-{int(df['season'].max())} "
            f"({dropped} rows dropped for missing features). R^2 {res.rsquared:.3f}, adjusted {res.rsquared_adj:.3f}, "
            f"residual SD {np.sqrt(res.scale):.2f} points. 95% intervals use couple-clustered standard errors.",
            "",
            "**In plain language** (each holds every other predictor fixed):",
            "",
            *[f"- {s}" for s in sentences],
            "",
            "### Predictors",
            "",
            "`effect`: change in the average judge score per unit of the predictor (binary/0-1 predictors: from 0 to 1). "
            "`std_effect`: effect of a one-standard-deviation change. `swing`: effect across `swing_range` "
            "(10th-90th percentile for continuous predictors, 0 to 1 otherwise).",
            "",
            md_table(pred[["term", "reading", "effect", "lo", "hi", "p", "std_effect", "swing", "swing_range"]]),
            "",
            "### Dance styles (vs the average dance)",
            "",
            "Each style's coefficient re-centered on the performance-weighted average of all styles, so 0 = an average dance. "
            f"`{OTHER}` collects the dances outside the one-hot list.",
            "",
            md_table(sty[["term", "effect", "lo", "hi", "p", "share_of_dances"]], pct={"share_of_dances"}),
            "",
            "### Which predictors matter most",
            "",
            "`r2_alone`: R^2 of the group on its own. `r2_drop`: R^2 lost when the group is removed. "
            "`shapley_r2`: the group's average contribution over every order of adding groups (sums to the full R^2).",
            "",
            md_table(imp, pct={"share_of_r2"}),
            "",
            "### Collinearity (VIF)",
            "",
            (f"VIF above 5: {', '.join(f'{t} ({v:.1f})' for t, v in zip(high_vif['term'], high_vif['vif']))}. "
             "Read these terms together (see the `form` row above), not one at a time."
             if len(high_vif) else "No VIF above 5."),
            "",
            md_table(vif.sort_values("vif", ascending=False), digits=2),
            "",
            "### OLS vs the production Ridge",
            "",
            f"Ridge (alpha {alphas[set_name]:g} on standardized features) refit on the same rows, coefficients converted to raw units. "
            "Small differences mean the OLS readings describe the production model. Ridge shrinkage moves weight between the "
            "collinear `prev_score` and `running_mean`, so compare their sum (`form`).",
            "",
            md_table(rvo),
            "",
            "### Robustness",
            "",
            f"Raw coefficients by era and with season fixed effects (absorbs season-to-season scoring inflation). "
            "Style columns here are vs the omitted `other` styles, not vs the average dance.",
            "",
            md_table(rob),
            "",
            "Clustered vs heteroskedasticity-robust (HC1) standard errors:",
            "",
            md_table(se),
            "",
        ]

    # Side-by-side comparison of the two models' effects.
    cmp = None
    for label, r in results.items():
        e = r["eff"][["term", "effect"]].rename(columns={"effect": label})
        cmp = e if cmp is None else cmp.merge(e, on="term", how="outer")
    order = [t for t in results["order-announced"]["eff"]["term"]]
    cmp = cmp.set_index("term").loc[order].reset_index()
    cmp.to_csv(out_dir / "effects_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharex=True)
    for ax, (label, r) in zip(axes, results.items()):
        forest(ax, r["eff"], f"{label} (`{r['model']}`)")
    fig.tight_layout()
    fig.savefig(out_dir / "effects_forest.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    imp_all = pd.concat([r["imp"].assign(model=label) for label, r in results.items()])
    piv = imp_all.pivot(index="group", columns="model", values="shapley_r2").fillna(0)
    piv = piv.loc[piv.max(axis=1).sort_values().index]
    piv.plot.barh(ax=ax)
    ax.set(xlabel="Shapley share of R^2 (points of R^2)", ylabel="", title="Predictor groups: contribution to explained variance")
    fig.tight_layout()
    fig.savefig(out_dir / "group_importance.png", dpi=110)
    plt.close(fig)

    lines = [
        "# OLS effects report: what moves the average judge score",
        "",
        "Generated by `scripts/report_ols.py`. OLS fit on all seasons with exactly the predictors of the two production "
        "Ridge models. The target is the couple's weekly average judge score (per judge, per dance, 1-10 scale). "
        "Coefficients are associations holding the other predictors fixed, not causal effects. Standard errors are "
        "clustered by couple because each couple appears once per week.",
        "",
        "![effects](effects_forest.png)",
        "",
        "![importance](group_importance.png)",
        "",
        "## Pre-show vs order-announced effects",
        "",
        "Plain-unit effects from both fits (styles vs the average dance). Adding the running order barely moves the other effects "
        "if the rows agree.",
        "",
        md_table(cmp),
        "",
        *sections,
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(md_table(cmp))
    print(f"\nSaved {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
