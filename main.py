"""gta6-twitch-ccu-forecast-01 entry: comparables-pooled quantile forecaster.

Reads each case in the input directory and, for hours 2 to 24 of the launch
day, writes five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's Twitch
viewer count in the case's category.

Model: every comparable in the case (a past big launch with an hourly curve
from 14 days before to 3 days after its release day) contributes its launch
day shape as log-ratios to its own hour-1 count, lr_c(h) = ln(v_c(h)/v_c(1)).
The forecast for hour h is the similarity-weighted empirical quantile of
those log-ratios, applied to this case's hour-1 print p1:

    q_tau(h) = p1 * exp(quantile_tau({lr_c(h)}))

Weights favour comparables that resemble the case's launch: the same
rolling-midnight unlock pattern, the same console-only / PC platform split,
a similar hour-1 scale (log-distance kernel), and a small penalty for
early-access-contaminated curves and for "cold" fixed-time unlocks (hour 1
near zero) when the case itself is a rolling console launch. The pooled
quantiles are narrowed by a spread factor of 0.70 around the median, and
the two outer quantiles (5% and 95%) get an extra additive pad of 0.07 in
log space on top of that: a second leave-one-out pass over the pool showed
the narrowed tails were too sharp (90% intervals covered only about 81% of
realised hours), and the pad lifts coverage to about 85% while slightly
lowering the loss. Both the weights and the calibration were chosen by
leave-one-out validation over the comparables pool (each comparable
forecast from the others), where this model's mean pinball loss on the log
scale is about 0.39 against about 0.93 for the persistence baseline below
(console-only subset: about 0.19 against about 0.32).

Fallback: if a case carries no usable comparable, the forecast is the house
persistence shape: every hour stays at hour 1's print with a log spread of
0.15 * sqrt(h - 1).

Standard library only, no network, deterministic: the output is a pure
function of the input file. Runs as `python3 /bot/main.py` with the case in
/input, or with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR set.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader runs with -P, which keeps this file's directory off sys.path;
# put it back first so a sibling module would import (this entry is one file).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
BASE_SIGMA = 0.15

# Calibration, chosen by leave-one-out validation over the comparables pool.
SCALE_S = 1.0    # log-distance kernel width for the hour-1 scale weight
ROLL_F = 2.5     # weight multiplier when the unlock pattern matches
CONS_F = 4.0     # weight multiplier when the platform split matches
COLD_F = 0.25    # weight multiplier for cold fixed-unlock curves (see below)
COLD_HOT = 0.02  # hour-1 / day-peak below this marks a cold unlock curve
SPREAD = 0.70    # pooled-quantile narrowing around the median, in log space
TAIL_PAD = 0.07  # extra additive widening of the 5%/95% quantiles, log space


def parse_ts(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def day_curve(comp):
    """The comparable's release-day curve {hour 1..24: viewers}, or None."""
    try:
        ds = parse_ts(comp["day_start"])
        day = {}
        for point in comp["hourly"]:
            off = (parse_ts(point["from"]) - ds).total_seconds() / 3600.0
            if 0 <= off < 24 and off == int(off):
                day[int(off) + 1] = point["viewers"]
    except (KeyError, TypeError, ValueError):
        return None
    return day if len(day) == 24 else None


def comp_features(comp, curve):
    plat = comp.get("platforms", "")
    unl = comp.get("unlock", "").lower()
    v1 = max(curve[1], 1)
    peak = max(max(curve.values()), 1)
    return {
        "has_pc": bool(re.search(r"\bPC\b", plat)) or "Stadia" in plat,
        "rolling": ("midnight local" in unl) or ("rolling" in unl),
        "v1": v1,
        "hot": curve[1] / peak,
        "lr": [math.log(max(curve[h], 1) / v1) for h in range(1, 25)],
    }


def target_features(case):
    """The case launch's profile, read from its notes: rolling-midnight
    unlock, and whether a PC version launched the same day."""
    notes = case.get("notes", "")
    t = notes.lower()
    rolling = ("rolling" in t) or ("midnight local" in t)
    pc_launch = (
        bool(re.search(r"\bpc\b", t))
        and "console only" not in t
        and not re.search(r"\bon (ps5|ps4|playstation|xbox|switch)[^.;]{0,40}\bonly\b", t)
    )
    return {"rolling": rolling, "console_only": not pc_launch}


def weight(comp, target, p1):
    w = 1.0
    if comp["rolling"] == target["rolling"]:
        w *= ROLL_F
    if (not comp["has_pc"]) == target["console_only"]:
        w *= CONS_F
    if target["rolling"] and target["console_only"] and comp["hot"] < COLD_HOT:
        w *= COLD_F
    w *= math.exp(-abs(math.log(comp["v1"] / p1)) / SCALE_S)
    return w


def weighted_quantile(pairs, tau):
    """Weighted quantile by midpoint-CDF linear interpolation."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    pts = []
    cw = 0.0
    for x, w in pairs:
        cw += w
        pts.append(((cw - w / 2.0) / total, x))
    if tau <= pts[0][0]:
        return pts[0][1]
    if tau >= pts[-1][0]:
        return pts[-1][1]
    for i in range(1, len(pts)):
        if pts[i][0] >= tau:
            (p0, x0), (p1_, x1) = pts[i - 1], pts[i]
            if p1_ == p0:
                return x1
            return x0 + (x1 - x0) * (tau - p0) / (p1_ - p0)
    return pts[-1][1]


def round2(x):
    """Round to 2 decimals, half up, as the starter's output does."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def baseline_row(p1, h):
    return [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1))) for z in Z]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    hours = [row["hour"] for row in case["forecast"]]

    pool = []
    for comp in case.get("comparables") or []:
        if comp.get("name") == case.get("game", {}).get("name"):
            continue
        curve = day_curve(comp)
        if curve is None or curve[1] < 1:
            continue
        pool.append(comp_features(comp, curve))

    if not pool:
        return {"hours": [{"hour": h, "q": baseline_row(p1, h)} for h in hours]}

    target = target_features(case)
    weighted = [(c, weight(c, target, p1)) for c in pool]
    rows = []
    for h in hours:
        pairs = [(c["lr"][h - 1], w) for c, w in weighted]
        qs = [weighted_quantile(pairs, tau) for tau in LEVELS]
        med = qs[2]
        vals = []
        for i, q in enumerate(qs):
            dev = (q - med) * SPREAD
            if i == 0:  # outer quantiles get an additive pad on top of the
                dev -= TAIL_PAD  # narrowed deviation: the pooled tails proved
            elif i == len(qs) - 1:  # too sharp for the biggest launches
                dev += TAIL_PAD
            v = p1 * math.exp(med + dev)
            vals.append(round2(min(max(v, 0.01), 1e9)))
        for i in range(1, len(vals)):  # keep the row non-decreasing
            if vals[i] < vals[i - 1]:
                vals[i] = vals[i - 1]
        rows.append({"hour": h, "q": vals})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
