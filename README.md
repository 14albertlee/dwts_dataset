# Dancing with the Stars Dataset

A dataset of Dancing with the Stars scraped from wikipedia tables.  Created for a Data Engineering class.

## Understanding the dataset

The database can be understood as:

> A `score` is awarded by single `judge` to a single `performance` and is out of 10. A `performance` is by a single `couple` made up of a `celebrity` and a `professional`, performing in a `primary_dance_style` set to a `primary_song` by a `primary_song_artist`. The performances take place during a `performances.week` of a particular `week_theme` which make up a `performances.season`.

The dataset currently covers seasons 1 through 35.

`couples` links a `couple` to its `celebrity_id` and `professional_id`. `performances` links a performance to its `couple_id` (in addition to `primary_dance_style_id`, `week_theme_id`, and `primary_song_id`). `performances` does not carry its own `celebrity_id`/`professional_id`; join through `couples` to get those, since `couple_id` already implies both.

`performances.eliminated` is a nullable boolean derived from Wikipedia's per-dance `result` column (`Safe`, `Eliminated`, `Bottom two`, `Runners-up`, etc.). It's resolved per (season, week, couple) rather than per dance, since a couple can have more than one dance (and `result` tag) on the same night — e.g. a team-challenge `Winners`/`Losers` tag alongside their regular dance, or a `Bottom two` jeopardy tag followed by the dance-off's actual verdict. The more elimination-definitive tag wins (so `Eliminated` beats a same-night `Losers`, and a dance-off's `Safe` beats an earlier `Bottom two`). `Runners-up` and the finale's `Third place`/`Fourth place`/`Fifth place` placements count as eliminated (they didn't win); `Winners`/`Losers` on team-challenge nights do not (those are about that night's challenge, not elimination). `eliminated` is left `NULL` for couple/weeks where no row carried a recognizable result tag at all (mostly premiere weeks, before eliminations start that season).

Currently performances by multiple couples at the same time are dropped. This removes the complication of different kinds of judgements, including judgements in which the judges rank the couples (rather than giving scores out of 10).

In some weeks judges give two scores each to a dance (a performance score and a technical score). Those are collapsed to a single score by averaging.

Sometimes performances are medleys with multiple dance styles and multiple songs.  Currently only the first listed dance_style and song are included (hence `primary`).  

A handful of weeks (e.g. season 23 week 5, season 31 week 6) list a different number of individual judge scores than judges named in that week's judge-panel sentence (substitute/guest judges, unusually formatted scores, etc); those rows are dropped from `scores.csv` rather than risk misaligning a judge with the wrong score. As of late 2025 Wikipedia's season 15 article no longer includes its judge-panel sentence anywhere on the page, so that season's judge names (a fixed panel of Carrie Ann Inaba, Len Goodman, and Bruno Tonioli all season, with no guest judges) are hardcoded in the scraper rather than parsed from page text.

I have not tried to standardize the type of `notability` for which a star is known. This would also need to be a hierarchy but is currently just a single column `celebrities.notability`.

Celebrities are teamed with a professional for the season. That enables professionals to claim wins over time. However sometimes the celebrity dances with a different professional for some performances, causes include substitutions for illness/injury, but sometimes there is a "switch up" and all celebrities dance with the professional. Note that the `couples` table is the couple listed in the weekly scores tables on the wikipedia pages, which reflects the actual people performing in that particular dance (as does the `performances.professional` column which is derived from the couple column on the wikipedia page.). The database currently does not have the season long pairings; this could be derived from the `performances` by finding the most frequent professionals with which a `celebrity` performed.

I don't think that the `ids` are stable through runs of `parse_scores_json_to_tables.py` because they are created with `pd.factorize` which I don't think is stable.  Or perhaps it is the spidering order and thus the order in the scores.json.  Either way the ids of things like judges change when you re-generate the dataset (they are consistent within each dataset, though!).

The scraper fetches pages live from Wikipedia on each crawl (scrapy's httpcache is enabled locally for speed during a single run, but `.scrapy/httpcache` is not committed to the repo, since Wikipedia content changes over time and a committed cache stops being a faithful snapshot). Expect the resulting dataset to drift slightly between runs as Wikipedia articles are edited.

The scraper produces `dwts_scrapy/scores.json` the actual scraping code is in `dwts_scraper/spiders/dwts_scores.py` and can be run with:

```sh
cd dwts_scrapy
scrapy crawl dwts_scores -O scores.json
```

There is then a parser script, `scripts/parse_scores_json_to_tables.py`, which reads `dwts_scrapy/scores.json` and (re)writes the output csv files in `data/raw/`. Run it from the repository root:

```sh
python scripts/parse_scores_json_to_tables.py
```

Within `data/raw/` each `csv` is a table. Primary keys are the first column (e.g., `performance_id`). Foreign keys are named with `<table_name>_id` (e.g., primary_dance_style_id).

### Querying the dataset with DuckDB

`scripts/build_duckdb.py` loads all the csvs in `data/raw/` into a single queryable [DuckDB](https://duckdb.org/) database file, `data/database/dwts_dataset.duckdb`, with the primary/foreign keys above declared as real `PRIMARY KEY`/`FOREIGN KEY` constraints (so loading it also double-checks the referential integrity of the csvs). Run it from the repository root after regenerating the csvs:

```sh
python scripts/build_duckdb.py
```

Then query it with the `duckdb` CLI or Python, e.g.:

```sh
duckdb data/database/dwts_dataset.duckdb -c "
SELECT celebrity, avg(judge_score) AS avg_score
FROM scores
JOIN performances USING (performance_id)
JOIN couples USING (couple_id)
JOIN celebrities USING (celebrity_id)
GROUP BY celebrity
ORDER BY avg_score DESC
LIMIT 5;
"
```

`data/database/dwts_dataset.duckdb` itself is not committed to the repo (it's a regeneratable build artifact, like the csvs it's built from); run `scripts/build_duckdb.py` locally to get one.

Note that this is set up as a repo that can work with `repo2docker`.  You can run it direcltly on mybinder via this link [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/howisonlab/dwts_dataset.git/HEAD)

Or you can get it running if you have Docker by installing `repo2docker` and running it.

```sh
brew install repo2docker
git clone https://github.com/howisonlab/dwts_dataset.git
cd dwts_dataset
repo2docker --volume .:on-disk .
```

That gives you a notebook environment in which you can edit the files in the `~/on-disk` folder and those edits will persist.
