# GTA VI Twitch CCU Forecaster

[![make.fun — gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/acengrahmi/makefun-gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/acengrahmi/makefun-gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Forecasts Grand Theft Auto VI's launch-day Twitch viewers hour by hour, from
the first hour's print alone. For each of hours 2–24 of 2026-11-19 (UTC) it
writes five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's concurrent
viewers in the game's Twitch category.

**Method** (Python 3.13, standard library only, no network, deterministic):
anchor on the hour-1 print; learn the launch-day shape from the case's own
comparable past launches — a 60/40 log-space ensemble of per-hour ridge
regression on three pre-midnight features and a k-nearest-neighbour median
shape — with quantile spreads calibrated from cross-validated residuals.
See REPORT.md for the full design and measured validation.

**Validation** (practice cases p00–p03, mean pinball loss on the log scale):
**0.1862** vs house baseline 0.3871 (top-10 dev cutoff 0.225, dev #1 0.169).
Runtime ~0.09 s per case; two runs byte-identical.

**Run it:**

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Entry for the [make.fun](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
competition `gta6-twitch-ccu-forecast-01`.
