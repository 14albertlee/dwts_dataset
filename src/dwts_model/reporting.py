"""Helpers shared by the report scripts (scripts/report_*.py)."""
import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, digits: int = 3, pct=()) -> str:
    """Markdown table of `df`; floats get `digits` decimals, columns in `pct` are shown as percentages."""
    def fmt(col, v):
        if v is None or v is pd.NA:
            return ""
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return f"{v:.0%}" if col in pct else f"{v:.{digits}f}"
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(c, v) for c, v in zip(cols, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
