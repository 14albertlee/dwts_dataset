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


# Celebrity background by likely dance ability, not fame type (3.4). Ordered: first match wins,
# so "Glee actress & dancer" is movement_trained and "Actor & singer" is musician.
CELEB_RULES: list[tuple[str, str]] = [
    ("movement_trained",
     r"gymnast|figure skater|ice dancer|dancer|broadway|cheer|las vegas entertainer|high school musical|"
     r"wwe|wrestler|peloton|yoga|fitness|"
     # pop groups with choreographed stage shows
     r"nsync|backstreet|fifth harmony|big time rush|98 degrees|new kids|boyz ii men|\br5\b|spice girls|"
     r"pussycat dolls|pentatonix"),
    ("athlete_other",
     r"\bnfl\b|\bnba\b|\bmlb\b|\bnhl\b|olympic|paralympic|boxer|\bufc\b|mixed martial|tennis|swimmer|jockey|"
     r"race car|nascar|indy|snowboarder|rodeo|bull rider|soccer|basketball|bodybuilder|savannah bananas"),
    ("musician", r"singer|rapper|musician|violinist|frontman"),
    ("actor", r"actor|actress|comedian"),
    ("reality_social",
     r"reality|bachelor|cast member|social media|youtube|\bvine\b|jersey shore|contestant|\bmodel\b|supermodel|"
     r"jackass|love thy nader|miss usa"),
]
CELEB_DEFAULT = "other"  # hosts, journalists, chefs, politicians, business people
CELEB_GROUPS = [c for c, _ in CELEB_RULES] + [CELEB_DEFAULT]
CELEB_REFERENCE = "actor"  # largest group; omitted from the indicators


def celebrity_group(notability: str | None) -> str:
    if notability is None or pd.isna(notability):
        return CELEB_DEFAULT
    text = notability.lower()
    for group, pattern in CELEB_RULES:
        if re.search(pattern, text):
            return group
    return CELEB_DEFAULT


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
