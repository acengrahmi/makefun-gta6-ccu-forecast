"""Forecast launch-day Twitch viewers hour by hour (gta6-twitch-ccu-forecast-01).

Reads every <id>.in case in $MAKEFUN_INPUT_DIR (default /input) and writes
/output/<id>.out per case: for hours 2..24, five quantiles of that hour's
concurrent viewers in the game's Twitch category.

Method (everything is computed from the case file at run time; nothing is
hardcoded, and there is no network access):
  * Anchor on hour 1's print p1.
  * Learn the shape of a launch day from the case's own comparables: past
    big launches' hourly curves. Each comparable contributes its release-day
    log-ratio curve log(viewers_h / viewers_1) and three features measured
    before midnight UTC of its release day:
      - log(pre(-1) / p1): viewers in the hour just before midnight vs hour 1
      - log(premax / p1): the pre-release day's peak vs hour 1
      - log(p1): the scale of the launch itself
    Only "sane" comparables are used: ones whose pre-release peak is within
    1x-4.5x of hour 1, whose day peak is below 25x hour 1, and whose hour 1
    is at least 5,000 viewers. That drops simultaneous-global-unlock games
    (dead categories at midnight, ratios in the hundreds) and already-peaked
    early-access launches, keeping rolling-midnight console launches like
    the target.
  * Two predictors are blended 60/40 in log space:
      1. Per-hour ridge regression of the log-ratio on the three features.
      2. The median log-ratio curve of the 5 nearest comparables in
         standardised feature space.
  * Quantile spreads come from each predictor's residual dispersion
    (regression residuals, neighbour dispersion), averaged, with a floor.
    Spreads widen naturally with horizon because late hours vary more
    across launches.
  * Fallbacks: if no comparable is usable, fall back to a flat persistence
    forecast with a widening spread. Every output is clamped to (0, 1e9],
    quantiles are forced non-decreasing, and the program is fully
    deterministic (no randomness, stable sort orders).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
HOURS = list(range(2, 25))
LOG_H1_CENTER = 11.5
REG_WEIGHT = 0.6
KNN_K = 5
RIDGE_LAMBDA = 1.0
SPREAD_MULT = 0.85
SPREAD_FLOOR = 0.10


def _parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def _solve_linear(mat, vec):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(mat)
    aug = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col] / pv
                for k in range(col, n + 1):
                    aug[r][k] -= f * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def _ridge_fit(rows, targets, lam):
    """Ridge regression (intercept unpenalised). rows: list of feature lists."""
    p = len(rows[0])
    ata = [[0.0] * p for _ in range(p)]
    atb = [0.0] * p
    for xi, yi in zip(rows, targets):
        for i in range(p):
            atb[i] += xi[i] * yi
            for j in range(p):
                ata[i][j] += xi[i] * xi[j]
    for i in range(1, p):
        ata[i][i] += lam
    return _solve_linear(ata, atb)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _hourly_curve(hourly_rows, day_start):
    """Map release-day hourly rows to {hour 1..24: viewers}."""
    curve = {}
    for row in hourly_rows:
        viewers = row.get("viewers")
        if not isinstance(viewers, (int, float)) or not viewers > 0:
            continue
        try:
            t = _parse_ts(row["from"])
        except (KeyError, ValueError):
            continue
        delta = (t - day_start).total_seconds() / 3600.0
        if 0 <= delta < 24:
            curve[int(delta) + 1] = float(viewers)
    return curve


def _pre_curve(hourly_rows, day_start):
    """Map pre-release hourly rows to {offset -24..-1: viewers}."""
    pre = {}
    for row in hourly_rows:
        viewers = row.get("viewers")
        if not isinstance(viewers, (int, float)) or not viewers > 0:
            continue
        try:
            t = _parse_ts(row["from"])
        except (KeyError, ValueError):
            continue
        delta = (t - day_start).total_seconds() / 3600.0
        if -24 <= delta < 0:
            pre[int(delta)] = float(viewers)
    return pre


def _features(p1, pre):
    """The three launch features from the pre-midnight curve.

    prem1: viewers in the hour just before midnight (or the latest
    available pre-midnight hour); premax: the pre-release day's peak.
    """
    if pre:
        latest = max(pre)
        prem1 = pre.get(-1, pre[latest])
        premax = max(pre.values())
    else:
        prem1 = p1
        premax = p1
    return {
        "prem1": math.log(max(1.0, prem1) / p1),
        "premax": math.log(max(1.0, premax) / p1),
        "logh1": math.log(p1) - LOG_H1_CENTER,
    }


def _feature_vector(feats):
    return [1.0, feats["prem1"], feats["premax"], feats["logh1"]]


def _clamp_features(feats, train_feats):
    """Clamp each feature to the training range widened by 25% of its span,
    so a much bigger (or smaller) launch than any comparable cannot push the
    regression into wild extrapolation."""
    out = dict(feats)
    for key in ("prem1", "premax", "logh1"):
        vals = [f[key] for f in train_feats]
        lo, hi = min(vals), max(vals)
        span = hi - lo or 1.0
        out[key] = min(max(feats[key], lo - 0.25 * span), hi + 0.25 * span)
    return out


def _build_table(case):
    """Usable comparables: each with its release-day curve and features.

    A comparable is usable when it has a positive hour-1 print and at
    least 18 of the 24 release-day hours. The returned list is sorted by
    name so every downstream step is deterministic.
    """
    table = []
    for comp in case.get("comparables") or []:
        try:
            day_start = _parse_ts(comp["day_start"])
        except (KeyError, ValueError):
            continue
        curve = _hourly_curve(comp.get("hourly") or [], day_start)
        if 1 not in curve or len(curve) < 18:
            continue
        pre = _pre_curve(comp.get("hourly") or [], day_start)
        p1 = curve[1]
        feats = _features(p1, pre)
        table.append({
            "name": comp.get("name", ""),
            "curve": curve,
            "feats": feats,
            "peak_ratio": max(curve.values()) / p1,
        })
    table.sort(key=lambda c: c["name"])
    return table


def _sane(table):
    """Keep rolling-midnight-like launches: pre-release peak 1x-4.5x of
    hour 1, day peak below 25x hour 1, hour 1 at least 5,000 viewers."""
    keep = []
    for row in table:
        premax_ratio = math.exp(row["feats"]["premax"])
        if not (1.0 <= premax_ratio <= 4.5):
            continue
        if row["peak_ratio"] >= 25.0:
            continue
        if row["curve"][1] < 5000:
            continue
        keep.append(row)
    return keep


def _log_ratios(row):
    h1 = row["curve"][1]
    return {h: math.log(v / h1) for h, v in row["curve"].items() if h >= 2}


def _residual_scale(rows, shape):
    """Per-hour residual std of log-ratios around a shape (median curve)."""
    scale = {}
    for h in HOURS:
        res = []
        for row in rows:
            lr = _log_ratios(row)
            if h in lr:
                res.append(lr[h] - shape[h])
        if len(res) >= 5:
            mean = sum(res) / len(res)
            var = sum((r - mean) ** 2 for r in res) / (len(res) - 1)
            scale[h] = math.sqrt(var)
        else:
            scale[h] = 0.35
    return scale


def _median_shape(rows):
    shape = {}
    for h in HOURS:
        vals = []
        for row in rows:
            lr = _log_ratios(row)
            if h in lr:
                vals.append(lr[h])
        shape[h] = _median(vals) if vals else 0.0
    return shape


def _regression_predict(table, target_feats):
    """Per-hour ridge regression of log-ratio on the three features.

    Returns {hour: (median log-ratio, residual std)}.
    """
    train_feats = [row["feats"] for row in table]
    target = _clamp_features(target_feats, train_feats)
    x_target = _feature_vector(target)
    out = {}
    for h in HOURS:
        rows, ys = [], []
        for row in table:
            lr = _log_ratios(row)
            if h in lr:
                rows.append(_feature_vector(_clamp_features(row["feats"], train_feats)))
                ys.append(lr[h])
        if len(rows) < 6:
            out[h] = (0.0, 0.35)
            continue
        coef = _ridge_fit(rows, ys, RIDGE_LAMBDA)
        if coef is None:
            out[h] = (_median(ys), 0.35)
            continue
        med = _dot(coef, x_target)
        res = [y - _dot(coef, x) for x, y in zip(rows, ys)]
        dof = max(1, len(res) - len(coef))
        sd = math.sqrt(sum(r * r for r in res) / dof)
        out[h] = (med, max(SPREAD_FLOOR, sd * SPREAD_MULT))
    return out


def _knn_predict(table, target_feats, k=KNN_K):
    """Median log-ratio curve of the k nearest comparables in standardised
    feature space, plus their residual scale around that median."""
    train_feats = [row["feats"] for row in table]
    target = _clamp_features(target_feats, train_feats)
    keys = ("prem1", "premax", "logh1")
    vecs = []
    for row in table:
        f = _clamp_features(row["feats"], train_feats)
        vecs.append([f[key] for key in keys])
    p = len(keys)
    mu = [sum(v[i] for v in vecs) / len(vecs) for i in range(p)]
    sd = []
    for i in range(p):
        var = sum((v[i] - mu[i]) ** 2 for v in vecs) / len(vecs)
        sd.append(math.sqrt(var) or 1.0)
    xt = [target[key] for key in keys]
    ranked = []
    for idx, v in enumerate(vecs):
        dist = math.sqrt(sum(((v[i] - xt[i]) / sd[i]) ** 2 for i in range(p)))
        ranked.append((dist, table[idx]["name"], idx))
    ranked.sort()
    near = [table[idx] for _, _, idx in ranked[:k]]
    shape = _median_shape(near)
    scale = _residual_scale(near, shape)
    scale = {h: max(SPREAD_FLOOR, s * SPREAD_MULT) for h, s in scale.items()}
    return {h: (shape[h], scale[h]) for h in HOURS}


def _persistence_forecast(p1):
    """Flat fallback: every hour at hour 1's print, spread widening like a
    random walk in log space. Used only when no comparable is usable."""
    out = {}
    for h in HOURS:
        s = 0.15 * math.sqrt(h - 1)
        out[h] = [max(1.0, p1 * math.exp(z * s)) for z in Z]
    return out


def _quantiles_from_log(median_log, spread):
    """Five quantiles from a log-median and log-spread, cleaned up."""
    vals = []
    prev = 0.0
    for z in Z:
        q = math.exp(median_log + z * spread)
        if not (q > 0.0) or q != q or q == float("inf"):
            q = math.exp(median_log)
        q = min(q, 1e9)
        q = max(q, prev, 1.0)
        vals.append(round(q, 2))
        prev = vals[-1]
    # rounding can break monotonicity by a cent; repair from the top down
    for i in range(len(vals) - 2, -1, -1):
        if vals[i] > vals[i + 1]:
            vals[i] = vals[i + 1]
    return vals


def forecast_case(case):
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    try:
        day_start = _parse_ts(case["day_start"])
    except (KeyError, ValueError):
        day_start = None
    pre = {}
    if day_start is not None:
        pre = _pre_curve(case.get("history_hourly") or [], day_start)
    feats = _features(p1, pre)

    try:
        table = _sane(_build_table(case))
    except Exception:
        table = []
    if len(table) < 6:
        forecast = _persistence_forecast(p1)
        return {"hours": [{"hour": h, "q": forecast[h]} for h in HOURS]}

    try:
        reg = _regression_predict(table, feats)
        knn = _knn_predict(table, feats)
    except Exception:
        forecast = _persistence_forecast(p1)
        return {"hours": [{"hour": h, "q": forecast[h]} for h in HOURS]}
    log_p1 = math.log(p1)
    hours = []
    for h in HOURS:
        med_reg, sd_reg = reg[h]
        med_knn, sd_knn = knn[h]
        med_log = log_p1 + REG_WEIGHT * med_reg + (1.0 - REG_WEIGHT) * med_knn
        spread = max(SPREAD_FLOOR,
                     REG_WEIGHT * sd_reg + (1.0 - REG_WEIGHT) * sd_knn)
        hours.append({"hour": h, "q": _quantiles_from_log(med_log, spread)})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # Never leave a case without a valid file: a flat forecast at
            # 1 viewer is a legal output and beats a missing one.
            forecast = {"hours": [{"hour": h, "q": [1.0, 1.0, 1.0, 1.0, 1.0]}
                                  for h in HOURS]}
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
