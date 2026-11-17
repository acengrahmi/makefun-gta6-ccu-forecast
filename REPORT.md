# REPORT — gta6-twitch-ccu-forecast-01 entry

## Approach

Forecast the release day's hourly Twitch viewers from the case file alone
(stdlib only, no network, deterministic). Everything is computed at run
time from the case's comparables; nothing is hardcoded.

1. **Anchor on hour 1's print p1.** Every forecast is a log-ratio curve
   times p1.
2. **Sane-comparable filter.** From the case's comparables, keep only
   rolling-midnight-like launches: pre-release peak 1x–4.5x of hour 1, day
   peak below 25x hour 1, hour 1 ≥ 5,000 viewers. This drops
   simultaneous-global-unlock games (dead categories at midnight, peak
   ratios in the hundreds) and already-peaked early-access launches.
   Typically 22 of 38 comparables survive.
3. **Three features** measured before midnight UTC of release day:
   `log(pre(-1)/p1)` (viewers in the hour just before midnight vs hour 1),
   `log(premax/p1)` (pre-release day's peak vs hour 1), `log(p1)` (launch
   scale). These predict the day's ramp: across sane comparables,
   `log(pre(-1)/p1)` correlates +0.71 with `log(peak/hour 1)`.
4. **Ensemble, 60/40 in log space:**
   - per-hour ridge regression (λ=1) of the log-ratio on the three
     features;
   - median log-ratio curve of the 5 nearest comparables in standardised
     feature space.
   Features are clamped to the training range (widened 25%) so a much
   bigger launch than any comparable (GTA VI) cannot extrapolate wildly.
5. **Spreads** from each predictor's residual dispersion (regression
   residuals ×0.85, neighbour dispersion ×0.85), averaged, floor 0.10 in
   log units. Spreads widen with horizon because late hours vary more
   across launches.
6. **Guardrails:** fallback to a flat persistence forecast if fewer than
   6 sane comparables (or any model step throws); per-case try/except in
   `main()` so a bad case still yields a valid file; quantiles clamped to
   (0, 1e9], forced non-decreasing, rounded to 2 decimals.

## Validation

Pinball loss implements the RULES formula exactly
(`2·max(τ·d, (τ−1)·d)`, `d = ln y − ln q`, log scale, 5 levels × 23
hours). Practice answers (hours 2–24 of the 4 practice launches) were
fetched once from SullyGnome's public chart API named in
`input/NOTICE.txt` (4 requests); hour 1 of each fetched series matches
the case's print exactly. They live in `practice_answers.json` and are
used only by the validation scripts, never by `main.py`.

### Real practice cases (run through main.py as the grader runs it)

| case | loss | typical miss | 90% coverage | baseline loss |
|------|------|--------------|--------------|---------------|
| p00 Marvel's Wolverine | 0.2699 | 59.8% | 0.91 | 0.7436 |
| p01 Red Dead Redemption 2 | 0.1641 | 32.4% | 0.96 | 0.2868 |
| p02 God of War Ragnarök | 0.1287 | 20.7% | 1.00 | 0.2146 |
| p03 Marvel's Spider-Man 2 | 0.1823 | 30.4% | 0.83 | 0.3033 |
| **mean** | **0.1862** | — | — | **0.3871** |

Mean practice loss **0.1862** vs house baseline 0.3871, top-10 dev
cutoff 0.225, #1 dev entry 0.169. Target (≤0.20, ideally ≤0.19) met.

### Leave-one-comparable-out CV (main.py's own functions, sane set)

22 held-out launches: mean loss 0.2667, **median 0.2054**, p90 0.5204.
(CV over all 37 comparables including simultaneous-unlock extremes is
dominated by unpredictable 100–2000x ramps and is not representative of
a rolling-midnight console launch.)

### Final-like check

Re-scoring the practice cases with the other three practice launches
added to the training set — mimicking the final case, which holds all
42 comparables including the four practice launches — gives mean loss
**0.1826** (ensemble 60/40).

### Things tried and rejected

- Plain median log-ratio over all comparables: 0.3997 (worse than baseline).
- Subset median + similarity weights: 0.2393.
- Peak-anchored hybrid (predict day peak, use peak-normalised shape): 0.2083.
- Same-clock-time ratio model (release hour = R × pre-release same-clock
  hour): 0.2206–0.2416.
- Asymmetric empirical residual-quantile spreads: 0.1994–0.2106 (residual
  quantiles from 22 points are too noisy; symmetric normal spreads win).
- kNN-5 alone: 0.1864 on practice but k-choice is unstable (k6: 0.2279),
  so it enters only as 40% of the ensemble.

## Runtime and determinism

- ~0.08–0.09 s per case locally (limit 8 s); total run over 4 cases ~0.35 s.
- Two runs on the same cases produce byte-identical outputs
  (`validate.py` asserts this).
- Output shape asserted on every run: exactly 23 rows, hours 2–24,
  five finite non-decreasing quantiles each, 0 < q ≤ 1e9, file ≤ 65,536
  bytes (actual ~1.5 KB).
- Edge cases verified: no comparables, only insane comparables, no
  pre-release history, sparse pre-release hours, hour-1 print of 37 and
  of 2,000,000, and a case with no hour-1 print at all — all produce
  valid output files without crashing.

## Iteration log

1. **v1** — exploration: comparable curve shapes, unlock-pattern
   heterogeneity, feature search (`log(pre(-1)/p1)` +0.71 corr with peak
   ratio); plain-median and weighted-median baselines (0.40 / 0.24).
2. **v2** — ridge regression on three features + kNN-5 ensemble 60/40;
   practice mean 0.1862. Empirical-quantile spreads tested and rejected.
3. **v3** — robustness revisions: numeric type guards on hourly rows,
   try/except fallbacks to persistence (model step) and to a flat
   1-viewer file (whole case); re-validated with identical practice
   numbers (0.1862) and all edge cases passing.

## Files

- `main.py` — the entry (Python 3.13, standard library only).
- `validate.py` — runs `main.py` as the grader does and scores it
  (pinball, typical miss, coverage, shape, determinism, runtime).
- `experiments.py`, `common.py` — model comparison harness and helpers
  (LOO-CV, candidate models, feature analysis).
- `practice_answers.json` — fetched SullyGnome answers for the 4
  practice cases (validation data only).

## What could not be verified

- The final case (h00) itself: GTA VI has not released yet, so the
  transfer to a single unseen mega-launch cannot be measured directly.
  The practice mean (0.1862) and the final-like augmented check (0.1826)
  are the best available proxies, and the model was deliberately kept
  simple (3 features, ridge + kNN median) to transfer rather than
  overfit the 4 practice cases.
- The dev leaderboard's #1 entry (0.169) was not reproduced; the gap to
  it is concentrated in the single hardest practice case (Wolverine,
  0.2699), whose 9.09x day peak is an outlier no feature fully explains.
