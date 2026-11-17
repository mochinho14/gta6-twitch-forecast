# GTA VI Launch-Day Twitch Forecast

Entry for the make.fun bounty **gta6-twitch-ccu-forecast-01**: forecast the
Twitch viewer count in the Grand Theft Auto VI category for hours 2 to 24 of
its release day (2026-11-19, UTC), five quantiles per hour (5%, 25%, 50%,
75%, 95%), given everything recorded up to the end of hour 1.

## Run it

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` case in the input directory and writes `<id>.out` next to the
outputs: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`
with exactly 23 rows for hours 2 to 24. In the grader's sandbox the
environment variables are unset and it reads `/input` and writes `/output`.
The program is deterministic: two runs on the same case produce identical
bytes, and one case takes well under a second (limit: 8 s).

## The model

Comparables pooling. Each case ships with past big launches whose hourly
Twitch curves cover their release days. For every comparable with a complete
release-day curve, take the log-ratio of each hour to its own hour 1,
`lr_c(h) = ln(v_c(h) / v_c(1))`. The forecast for hour `h` is a
similarity-weighted empirical quantile of those log-ratios, applied to the
case's hour-1 print `p1`:

```
q_tau(h) = p1 * exp(quantile_tau({lr_c(h)}))
```

Weights reward comparables that resemble the case's launch (the case profile
is read from its notes): matching rolling-midnight unlock pattern (x2),
matching console-only vs PC platform split (x2), a log-distance kernel on the
hour-1 scale (`exp(-|ln(v1_c / p1)|)`), and a x0.25 penalty on "cold"
fixed-time unlocks (hour 1 below 2% of their day peak) when the case is a
rolling console launch. The pooled quantiles are narrowed by a spread factor
of 0.75 around the median. If a case has no usable comparable, the program
falls back to the persistence shape (flat at hour 1, log spread
`0.15 * sqrt(h - 1)`).

## Validation

Leave-one-out over the comparables pool: each of the 37 usable comparables
was forecast from the other 36 and scored with the contest's mean pinball
loss on the log scale.

| model | LOO loss (all 37) | LOO loss (8 console-only) |
| --- | --- | --- |
| persistence baseline | 0.930 | 0.323 |
| this model | 0.397 | 0.192 |

The weights and the spread factor sit in the middle of the flat region of
the validation surface, not on a spike. (These are validation numbers on
past launches, not a practice or final score; the server grades the entry.)

## Layout

- `main.py`: the whole entry, one file.
- `LICENSE`: MIT (from the competition kit's starter).
