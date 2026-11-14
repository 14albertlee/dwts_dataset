"""Rule-based groupings of dance styles and week themes.

Themes are classified from their text with ordered keyword rules (first match
wins), so new or renamed themes are handled without maintaining an id table.
"""
import re

import pandas as pd

# Ordered: earlier categories win. Patterns are matched case-insensitively.
THEME_RULES: list[tuple[str, str]] = [
    ("Personal / Emotional", r"memorable year|best year|stor(y|ies)|dedication|hometown|personal|celebrity'?s pick"),
    ("Disney", r"disney"),
    ("Competition Stage", r"premiere|first dances|first elimination|semi-?finals?|quarter-?finals?|finals?|finale|\btop \d+\b|elimination|race to"),
    ("Milestone / Special", r"birthday|\d+(st|nd|rd|th).*episode|special"),
    ("Dance / Competition", r"ballroom|trio|switch-?up|team|face-?off|dance-?off|duels?|choreography|fusion|famous dances|iconic dances|side-by-side|marathon|showstoppers"),
    ("Movies / TV", r"movie|\btv\b|film|hollywood|oscars?|\bbond\b|grease|wicked|villains|horror|monster|halloween|cirque|broadway"),
    ("None", r"^no theme$"),
]
THEME_DEFAULT = "Music"

# Ordered easiest -> hardest, as judged for this show.
STYLE_DIFFICULTY_ORDER = [
    "Jazz", "Contemporary", "Charleston", "Foxtrot", "Cha-Cha-Cha", "Waltz", "Salsa",
    "Viennese Waltz", "Argentine Tango", "Tango", "Paso Doble", "Rumba", "Jive",
    "Quickstep", "Samba",
]
STYLE_DIFFICULTY = {s: i + 1 for i, s in enumerate(STYLE_DIFFICULTY_ORDER)}

LATIN = {"Cha-Cha-Cha", "Samba", "Rumba", "Paso Doble", "Jive", "Salsa", "Mambo", "Argentine Tango"}
BALLROOM = {"Foxtrot", "Quickstep", "Tango", "Viennese Waltz", "Waltz"}


def theme_category(theme: str | None) -> str:
    if theme is None or pd.isna(theme):
        return "None"
    text = theme.strip().lower()
    if re.fullmatch(THEME_RULES[-1][1], text):
        return "None"
    for category, pattern in THEME_RULES[:-1]:
        if re.search(pattern, text):
            return category
    return THEME_DEFAULT


def normalize_style(style: str | None) -> str:
    """Treat 'Instant X' improvisation versions as the base dance."""
    if style is None or pd.isna(style):
        return "Other"
    return re.sub(r"^Instant ", "", style)


def style_group(style: str | None) -> str:
    s = normalize_style(style)
    if s in LATIN:
        return "Latin"
    if s in BALLROOM:
        return "Ballroom"
    return "Other"


def style_difficulty(style: str | None) -> float:
    """Ordinal difficulty (1 = easiest); NaN for dances outside the ranked list."""
    return float(STYLE_DIFFICULTY.get(normalize_style(style), float("nan")))


# Dances common enough (100+ performances) for their own indicator; the rest fall into "other".
ONEHOT_STYLES = [
    "Cha-Cha-Cha", "Foxtrot", "Quickstep", "Jive", "Paso Doble", "Tango", "Viennese Waltz",
    "Samba", "Rumba", "Argentine Tango", "Salsa", "Jazz", "Freestyle", "Contemporary", "Waltz",
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
