# Prompt: upcoming episode schedule CSV

Copy everything below the line into Claude. Save the CSV it returns as `schedule.csv` and run
`python scripts/predict_upcoming.py --schedule schedule.csv`.

---

Find the theme and each remaining couple's dances for the next upcoming episode of **Dancing with
the Stars (US)** (the next one that has not aired yet). Return a CSV in exactly this format, in a
single code block, with nothing else in the block:

```csv
celebrity,professional,theme,styles,order
Ezra Frech,Daniella Karagach,Semifinal,Quickstep; Jive,
Harry Shum Jr.,Jenna Johnson,Semifinal,Quickstep; Paso Doble,
```

Rules:
- One row per couple still competing. Use the celebrity's full name.
- `theme`: the episode's theme name (e.g. `Disney Night`), the same value on every row.
- `styles`: every dance the couple performs that night, in the order they perform them, separated by
  `; `. Leave out team or group dances shared by several couples. Use standard names: `Cha-Cha-Cha`,
  `Foxtrot`, `Quickstep`, `Jive`, `Paso Doble`, `Tango`, `Viennese Waltz`, `Samba`, `Rumba`,
  `Argentine Tango`, `Salsa`, `Jazz`, `Freestyle`, `Contemporary`, `Waltz`, `Charleston`,
  `Hustle`, `Lindy Hop`.
- `order`: leave empty. I fill it in when the running order is announced at the start of the
  episode.
- If a couple's dance is not announced yet, write `TBD` in `styles`.
- Below the code block, state the season, week and air date of the episode, and flag any uncertain
  or `TBD` entries.

When the order is announced, fill `order` with the couple's 1-based position in each dance, in the
same `;` layout as `styles` (e.g. `3; 2` = 3rd in the first round of dances, 2nd in the second).
Fill it for every couple or for none.
