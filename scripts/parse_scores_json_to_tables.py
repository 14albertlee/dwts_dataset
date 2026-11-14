"""Parse dwts_scrapy/scores.json into the normalized tables in data/raw/.

This is a straight-line script conversion of the original
`parse_scores_json_to_tables.ipynb` notebook (see git history for the
notebook if you need the exploratory/commented-out cells). Run it from the
repository root:

    python scripts/parse_scores_json_to_tables.py

It reads `dwts_scrapy/scores.json` and (re)writes every csv under
`data/raw/`.
"""

import csv
import os

import numpy as np
import pandas as pd
from pluralizer import Pluralizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "raw")
SCORES_JSON = os.path.join(REPO_ROOT, "dwts_scrapy", "scores.json")


def load_scores(path):
    """Load the raw scraped performances and do the initial column cleanup."""
    df = (
        pd.read_json(path)
        # scores comes in as one row per performance
        .assign(performance_id=lambda df_: np.arange(df_.shape[0]))
        .replace(r'^\s*$', np.nan, regex=True)
        .rename(columns={'judge': 'judge_excluded', "dance": "dance_style", "dance_&_theme": "dance_and_theme"})
        .dropna(subset=['couple'])
        .assign(judge=lambda df_: df_.judge_phrase.str.strip(to_strip=" .\n").str.split(", ", expand=False))
        .assign(week=lambda df_: df_.week_title.str.extract(r'Week (\d+)'),  # NaN if no Week Theme.
                week_theme=lambda df_: df_.week_title.str.extract(r': (.*?)$').fillna('No Theme'))
        # "Caveman" Hustle -> Hustle
        .assign(dance_style=lambda df_: np.where(
            df_.dance_and_theme.notna(),
            df_.dance_and_theme.str.extract(r'(?:".*?")?(.*?)$')[0],
            df_.dance_style,
        ))
        .assign(dance_styles=lambda df_: df_.dance_style.str.split("---"),
                primary_dance_style=lambda df_: df_.dance_styles.apply(lambda x: x[0] if isinstance(x, list) else np.nan))
        .assign(primary_dance_style=lambda df_: df_.primary_dance_style.str.replace("&", '').str.strip().str.title())
    )
    return df


def coalesce_music_columns(df):
    """Fold all the special-week "<theme>_music" columns into one `music` column."""
    # Maps each special-week music column to the artist suffix to append (or
    # None if the column text already stands on its own).
    music_col_artists = {
        'elvis_music': 'Elvis Presley',
        'classical_music': None,
        'motown_music': None,
        'stevie_wonder_music': 'Stevie Wonder',
        'music_by_britney_spears': 'Brittany Spears',
        'music_from_grease': None,
        'cher_music': 'Cher',
        'music_by_queen': 'Queen',
        'music_by_janet_jackson': 'Janet Jackson',
        'michael_buble_music': 'Michael Buble',
    }

    for col, artist in music_col_artists.items():
        if col not in df.columns:
            continue
        values = df[col] + f"—{artist}" if artist else df[col]
        df['music'] = np.where(df[col].notna(), values, df['music'])

    df = (
        df
        .drop(columns=[c for c in music_col_artists if c in df.columns])
        .assign(songs=lambda df_: df_.music.str.split("---"),
                primary_song_phrase=lambda df_: df_.songs.apply(lambda x: x[0] if isinstance(x, list) else np.nan))
    )
    df[['primary_song', 'primary_song_artist']] = df.primary_song_phrase.str.extract(r'(.*)[—―一]+(.*?)$')
    return df


def split_double_scores(df):
    """Some weeks give each couple a technical score and a performance score
    instead of a single score; split those into two rows with a score_type
    column, and give everything else a single "single_score" score_type."""
    double_scores = (
        df[df['technical_scores'].notna()]
        .drop(columns="scores")
        .dropna(axis=1, how="all")
    )
    val_cols = ['technical_scores', 'performance_scores']
    id_cols = [ele for ele in double_scores if ele not in val_cols]
    double_scores = double_scores.melt(id_vars=id_cols, value_vars=val_cols, value_name="scores", var_name="score_type")

    single_scores = df.drop(df.index[df['technical_scores'].notna()])
    single_scores = single_scores.copy()
    single_scores['score_type'] = "single_score"
    single_scores.drop(columns=['technical_scores', 'performance_scores'], inplace=True)

    return pd.concat([double_scores, single_scores])


def explode_judge_scores(df_score_type):
    """Turn "total (j1, j2, j3)" scores strings into one row per judge score,
    dropping multi-couple dances and otherwise-unparseable rows."""
    df_score_type[['total_score', 'indiv_scores']] = (
        df_score_type['scores'].str.extract(r'(?P<total_score>\d+)\s\((?P<indiv_scores>.*)\)')
    )

    # most are multi couple dances, with ranking scores.
    df_score_type["couple_list"] = df_score_type["couple"].str.split("---")
    df_score_type["couple_count"] = df_score_type["couple_list"].apply(len)

    # Now eliminate multi-couple dances and dances with NA total scores
    # (mostly dance offs and others). Might still be one or two issues with
    # total scores but not individual scores.
    df_single_couple = df_score_type.loc[
        (df_score_type['couple_count'] == 1) & df_score_type['total_score'].notna()
    ].copy()

    df_single_couple['judge_score_list'] = df_single_couple['indiv_scores'].str.split(", ")
    df_single_couple['judge_score_count'] = df_single_couple['judge_score_list'].apply(len)
    df_single_couple['judge_count'] = df_single_couple['judge'].apply(len)

    # A handful of weeks (e.g. season 23 week 5, season 31 week 6) list a
    # different number of individual judge scores than judges in the judge
    # panel sentence (substitute/guest judges not mentioned in the intro
    # paragraph, scores formatted unusually, etc). explode() requires the
    # list columns to have matching lengths, so drop those rows rather than
    # crash/misalign judge-to-score pairs; this mirrors the other "drop
    # unparseable edge cases" filters above.
    mismatched = df_single_couple['judge_score_count'] != df_single_couple['judge_count']
    if mismatched.any():
        bad_rows = df_single_couple.loc[mismatched, ['season', 'week_title', 'couple']]
        print(f"Dropping {mismatched.sum()} row(s) with mismatched judge/score counts:\n{bad_rows}")
    df_single_couple = df_single_couple.loc[~mismatched]

    df_final = (
        df_single_couple
        .explode(['judge', 'judge_score_list'])
        .rename(columns={'judge_score_list': 'judge_score'})
        .drop(columns=['indiv_scores', 'judge_phrase', 'scores'])
        .assign(judge=lambda df: df.judge.str.strip(" ."))
    )

    df_final.drop(df_final.query(" judge_score == 'X' ").index, inplace=True)
    df_final['judge_score_id'] = np.arange(df_final.shape[0])

    # rename "Guest judge" with the value from the guest_judge column.
    df_final['judge'] = np.where(df_final['judge'] == 'Guest judge', df_final['guest_judge'], df_final['judge'])
    # fix Julianne Hough (Night 1 only)
    df_final['judge'] = np.where(
        df_final['judge'] == 'Julianne Hough (Night 1 only)', 'Julianne Hough', df_final['judge']
    )

    # Strip stray Wikipedia footnote markers (e.g. "9[a]") that occasionally
    # leak into an individual judge's score cell.
    df_final['judge_score'] = (
        df_final['judge_score'].str.replace(r'\[.*?\]', '', regex=True).str.strip()
    )
    df_final['judge_score'] = df_final['judge_score'].astype(float)
    return df_final


def average_double_score_weeks(df_final):
    """Weeks with a technical/performance score produce two rows per judge
    that only differ by score_type; average those into a single judge_score
    and collapse back down to one row per (performance, judge)."""
    df_agg = df_final.groupby(["performance_id", "judge"]).agg({'judge_score': "mean"}).reset_index()

    df_summarized = (
        df_final
        .drop(columns=['score_type', 'judge_score'])
        .groupby(["performance_id", "judge"])
        .head(1)
        .merge(df_agg, how='left', on=["performance_id", "judge"])
    )
    return df_summarized


def move_to_front(df, front_cols):
    all_cols = df.columns.values.tolist()
    remaining_cols = [ele for ele in all_cols if ele not in front_cols]
    return df.reindex(columns=front_cols + remaining_cols)


def reorder_and_trim_columns(df_final):
    return (
        df_final
        .drop(columns=['judge_excluded', 'couple_list', 'couple_count', 'judge_score_count',
                       'judge_count', 'dance_styles', 'songs', 'primary_song_phrase', 'dance_and_theme'])
        .pipe(move_to_front, front_cols=['judge_score_id', 'judge', 'judge_score', 'performance_id',
                                         'season', 'week', 'week_theme', 'couple',
                                         'primary_dance_style', 'primary_song', 'primary_song_artist',
                                         'total_score'])
    )


def write_csv(df, name, pluralizer):
    filename = os.path.join(OUTPUT_DIR, pluralizer.plural(name) + ".csv")
    df.to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC)


def _canon_result(value):
    """Map a raw scraped `result` cell to a canonical status label, or None
    if it isn't a real result at all.

    A handful of `result` cells actually leaked an adjacent song/artist table
    cell (e.g. '"Hot Stuff" — Donna Summer') instead of a real result; those
    are recognizable because they start with a quote mark, and are treated
    as "no result" rather than a fabricated category. Everything else is
    normalized (e.g. the several 'Eliminated---...' / apostrophe-mangled
    variants all collapse to plain 'Eliminated').
    """
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value.startswith('"'):
        return None
    low = value.lower()
    if low.startswith('eliminated'):
        return 'Eliminated'
    if low.startswith('safe'):
        return 'Safe'
    if low == 'third place':
        return 'Third place'
    return value


# A couple can have more than one dance (and `result` cell) on the same
# night (e.g. a team-challenge dance plus their regular dance, or a
# "bottom two"/jeopardy announcement followed by a dance-off verdict). Those
# per-dance tags aren't contradictory, just differently final: pick
# whichever tag present for that (season, week, couple) is highest priority
# here, i.e. most decisive about whether the couple left the competition.
_RESULT_PRIORITY = [
    'Eliminated', 'Withdrew',
    'Third place', 'Fourth place', 'Fifth place',
    'Runners-up', 'Winners',
    'Bottom two', 'Bottom three', 'Bottom five',
    'Safe', 'Losers', 'Immunity',
]
_ELIMINATED_RESULTS = {
    'Eliminated', 'Withdrew', 'Third place', 'Fourth place', 'Fifth place', 'Runners-up',
}


def _resolve_result(categories):
    cats = set(c for c in categories if pd.notna(c))
    for status in _RESULT_PRIORITY:
        if status in cats:
            return status
    return None


def compute_eliminated(df_final):
    """Add an `eliminated` nullable-boolean column to df_final: whether the
    couple was eliminated after that week, derived from the scraped
    per-dance `result` text and resolved to one status per (season, week,
    couple_id) using `_resolve_result`. Left as <NA> for couple/weeks where
    no row carried a recognizable result (mostly premiere weeks, before any
    elimination has happened yet)."""
    result_category = df_final['result'].apply(_canon_result)
    resolved_status = (
        result_category
        .groupby([df_final['season'], df_final['week'], df_final['couple_id']])
        .transform(_resolve_result)
    )
    eliminated = resolved_status.isin(_ELIMINATED_RESULTS).astype('boolean')
    eliminated[resolved_status.isna()] = pd.NA
    df_final['eliminated'] = eliminated
    return df_final


def build_dimension_tables(df_final, pluralizer):
    """Factorize each catalog dimension into its own id, write out the
    dimension's csv, and add the id column back onto df_final so later
    tables (couples, performances) can use it as a foreign key."""
    # judges, celebrities and professionals all in different tables, although
    # they are all people and people do migrate between roles.
    #
    # Note: 'couple' is deliberately excluded from this generic loop. The
    # scraped 'couple' column is just a display label (e.g. "Chris &
    # Witney"), and the same label text recurs across seasons whenever a
    # different celebrity happens to share a first name with a previous one
    # partnered with the same pro (e.g. Chris Soules / Chris Kattan / Chris
    # Mazdzer, all "Chris & Witney"). Factorizing couple_id straight off that
    # text would incorrectly merge those distinct celebrity/pro pairs into
    # one couple_id. Instead couple_id is derived below from the actual
    # (celebrity_id, professional_id) pair once both ids exist.
    for dim in ['primary_dance_style', 'week_theme', 'judge', 'professional', 'primary_song_artist']:
        df_final[dim + "_id"] = df_final[dim].factorize()[0] + 1
        write_csv(df_final[[dim + "_id", dim]].drop_duplicates(), dim, pluralizer)

    # celebrity needs known_for (notability)
    df_final['celebrity_id'] = df_final.celebrity.factorize()[0] + 1
    write_csv(
        df_final.filter(['celebrity_id', "celebrity", 'notability'], axis='columns')
                .drop_duplicates(subset=['celebrity_id', 'celebrity']),
        'celebrity', pluralizer,
    )

    # couple_id is keyed off the real (celebrity_id, professional_id) pair,
    # not the display label text (see note above).
    df_final['couple_id'] = (
        df_final[['celebrity_id', 'professional_id']]
        .apply(tuple, axis=1)
        .factorize()[0] + 1
    )

    # elimination status is derived from couple_id, so couple_id must exist first.
    df_final = compute_eliminated(df_final)

    # song needs primary_artist
    df_final['primary_song_id'] = df_final.primary_song.factorize()[0] + 1
    write_csv(
        df_final.filter(['primary_song_id', "primary_song", 'primary_song_artist_id'], axis='columns')
                .drop_duplicates(subset=['primary_song_id', "primary_song"]),
        'primary_song', pluralizer,
    )

    return df_final


def write_couples_table(df_final, pluralizer):
    """couples.csv links each couple to the celebrity_id/professional_id that
    make it up, so couples can be joined back to celebrities/professionals."""
    couples = (
        df_final
        .filter(['couple_id', 'couple', 'celebrity_id', 'professional_id'], axis='columns')
        .drop_duplicates(subset=['couple_id'])
        .sort_values('couple_id')
    )
    write_csv(couples, 'couple', pluralizer)


def write_performances_table(df_final):
    # celebrity_id/professional_id are deliberately left off: couple_id
    # already implies both (via couples.celebrity_id/professional_id), so
    # carrying them here too would just be redundant, denormalized copies
    # that could drift out of sync with the couples table.
    (
        df_final
        .filter(['performance_id', 'season', 'week', 'week_theme_id',
                 'couple_id', 'primary_dance_style_id', 'primary_song_id',
                 'eliminated'], axis='columns')
        .drop_duplicates()
        .to_csv(os.path.join(OUTPUT_DIR, "performances.csv"), index=False, quoting=csv.QUOTE_NONNUMERIC)
    )


def write_scores_table(df_final):
    (
        df_final
        .assign(score_id=lambda df_: np.arange(1, df_.shape[0] + 1))
        .filter(['score_id', 'performance_id', 'judge_id', 'judge_score'], axis='columns')
        .to_csv(os.path.join(OUTPUT_DIR, "scores.csv"), index=False, quoting=csv.QUOTE_NONNUMERIC)
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_scores(SCORES_JSON)
    df = coalesce_music_columns(df)
    df_score_type = split_double_scores(df)
    df_final = explode_judge_scores(df_score_type)
    df_final = average_double_score_weeks(df_final)
    df_final = reorder_and_trim_columns(df_final)

    pluralizer = Pluralizer()
    df_final = build_dimension_tables(df_final, pluralizer)
    write_couples_table(df_final, pluralizer)
    write_performances_table(df_final)
    write_scores_table(df_final)


if __name__ == "__main__":
    main()
