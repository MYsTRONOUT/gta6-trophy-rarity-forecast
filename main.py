"""GTA VI trophy rarity forecast (Python 3.13, standard library only).

For every PlayStation 5 trophy of Grand Theft Auto VI, forecast its earn rate
on Exophase at the case's target, as five quantiles (5, 25, 50, 75, 95 percent)
of the rate in percent. The score is the mean pinball loss on the logit scale,
so the quantiles are the honest quantiles of a predictive distribution, not a
point guess with a decorative band.

How the forecast is made, per trophy:
  * The case history is every Exophase read before the checkpoint. A read gives
    the players tracked and each trophy's rate, so the earned count for a trophy
    is rate * players / 100. Two forces move a rate: players earning the trophy
    (earned counts rise as people play) and new players arriving (tracked counts
    rise, diluting every rate). The program models both, then divides.
  * Earned counts and tracked counts are each fitted in log space against
    ln(1 + hours since release), by weighted least squares that favours recent
    reads, and read at the target hour. The ratio is the median rate.
  * A direct fit of the logit's own trend is blended in half-and-half, so a
    trophy whose earned counts are too thin to fit still gets a sensible line.
  * The spread is a random walk in logit units that grows with the horizon,
    plus the fits' residual noise, plus a floor for the refresh risk Exophase
    carries (its counts jump in bursts, and a jump dilutes every rate at once).
    A long horizon therefore gets a wide, honest band instead of a confident
    miss.

Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json, math, os, sys
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

IN_DIR = pathlib.Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = pathlib.Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The five quantile levels, and the standard normal's quantile at each.
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZS = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

RELEASE = "2026-11-19T00:00:00Z"
# A rate is clamped before its logit, because 0% and 100% have no logit.
LO, HI = 0.01, 99.99
# Random-walk spread, in logit units per square-root day (the house baseline's).
WALK = 0.5
# Floor spread for refresh risk, in logit units: Exophase's counts jump in
# bursts and a jump dilutes every rate at once, so no forecast is ever narrow.
FLOOR_SIGMA = 1.1
# How far a trend may move a trophy's logit from its last read, in logit units.
MAX_MOVE = 3.5
# How much of the fitted move is trusted (the rest is pulled back to the last read).
DAMP = 0.7


def clamp_pct(p):
    return LO if p < LO else HI if p > HI else p


def to_logit(p):
    c = clamp_pct(p)
    return math.log(c / (100.0 - c))


def to_pct(l):
    if l >= 0:
        e = math.exp(-l)
        return 100.0 / (1.0 + e)
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; a whole number comes back as an int."""
    s = x * 10000.0
    w = math.floor(s)
    if s - w >= 0.5:
        w += 1
    v = w / 10000.0
    return int(v) if v == int(v) else v


def parse_time(text):
    """Seconds since 2026-11-19T00:00:00Z for an ISO timestamp ending in Z."""
    mo = int(text[5:7])
    d = int(text[8:10])
    h = int(text[11:13])
    mi = int(text[14:16])
    s = int(text[17:19])
    # Days since 2026-11-19: November counts from the 19th, December adds
    # November's remaining 11 days. Only those two months occur here.
    days = (11 + d) if mo == 12 else (d - 19)
    return days * 86400 + h * 3600 + mi * 60 + s


def wls(xs, ys, ws):
    """Weighted least squares of y on x. Returns (intercept, slope, resid_sd).

    Returns None when there are fewer than two distinct x values or the
    weights carry no information.
    """
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) * (x - mx) for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    inter = my - slope * mx
    n = len(xs)
    if n <= 2:
        resid = 0.0
    else:
        sse = sum(w * (y - (inter + slope * x)) ** 2 for x, y, w in zip(xs, ys, ws))
        resid = math.sqrt(max(0.0, sse) / max(1.0, (sw * (n - 2) / n)))
    return (inter, slope, resid)


def series_fit(hours, values, last_hour, target_hour):
    """Fit log(value) against ln(1+hours), read at the target. Returns
    (prediction at target, residual sd) or None when it cannot be fitted."""
    pts = [(h, v) for h, v in zip(hours, values) if v is not None and v > 0]
    if len(pts) < 3:
        return None
    xs = [math.log1p(h) for h, _ in pts]
    ys = [math.log(v) for _, v in pts]
    # Recent reads count more: a read a day old weighs about a third of a fresh one.
    ws = [math.exp(-(last_hour - h) / 24.0) for h, _ in pts]
    fit = wls(xs, ys, ws)
    if fit is None:
        return None
    inter, slope, resid = fit
    pred = inter + slope * math.log1p(target_hour)
    return (pred, resid)


def forecast_trophy(rates, tracked, hours, target_hour, horizon_days):
    """Five quantiles for one trophy. rates/track/hours are the history's
    per-read values, oldest first; a None rate means that read lacked it."""
    last_rate = None
    for r in reversed(rates):
        if r is not None:
            last_rate = r
            break
    if last_rate is None:
        last_rate = 1.0
    last_logit = to_logit(last_rate)
    last_hour = hours[-1] if hours else 0.0

    # Direct trend: the logit's own line against ln(1 + hours).
    direct = None
    pts = [(h, to_logit(r)) for h, r in zip(hours, rates) if r is not None]
    if len(pts) >= 3:
        xs = [math.log1p(h) for h, _ in pts]
        ys = [v for _, v in pts]
        ws = [math.exp(-(last_hour - h) / 24.0) for h, _ in pts]
        fit = wls(xs, ys, ws)
        if fit is not None:
            inter, slope, resid = fit
            direct = (inter + slope * math.log1p(target_hour), resid)

    # Structural trend: earned counts and tracked counts fitted apart, then divided.
    struct = None
    earned = []
    for r, t in zip(rates, tracked):
        if r is None or t is None:
            earned.append(None)
        else:
            earned.append(max(0.0, r / 100.0 * t))
    e_fit = series_fit(hours, earned, last_hour, target_hour)
    t_fit = series_fit(hours, tracked, last_hour, target_hour)
    if e_fit is not None and t_fit is not None:
        e_pred, e_res = e_fit
        t_pred, t_res = t_fit
        if t_pred > 0:
            ratio = math.exp(e_pred) / math.exp(t_pred) * 100.0
            struct = (to_logit(ratio), math.sqrt(e_res * e_res + t_res * t_res))

    # Blend the two medians in logit space; fall back to whichever fitted.
    cands = []
    if direct is not None:
        cands.append(direct)
    if struct is not None:
        cands.append(struct)
    if not cands:
        med = last_logit
        resid = 0.0
    else:
        med = sum(c[0] for c in cands) / len(cands)
        resid = max(c[1] for c in cands)
    # Damp the move from the last read and cap it: a week of play moves a
    # trophy a lot, but not without limit.
    move = med - last_logit
    move = max(-MAX_MOVE, min(MAX_MOVE, move * DAMP))
    med = last_logit + move
    # A trophy nobody has yet still gets earned by someone over a long week:
    # lift its median off the floor instead of insisting it stays at zero.
    if last_rate <= 0.05 and horizon_days >= 1.0:
        med = max(med, to_logit(0.2))

    # Spread: the walk over the horizon, the fits' noise scaled up with it,
    # and the refresh floor. Long horizons are honestly wide.
    var = (WALK * WALK * horizon_days) + (resid * resid * (1.0 + horizon_days / 7.0)) + FLOOR_SIGMA * FLOOR_SIGMA
    sigma = math.sqrt(max(0.05, var))
    qs = []
    for z in ZS:
        q = round4(clamp_pct(to_pct(med + z * sigma)))
        qs.append(q)
    # Rounding can only keep the order (z rises), but make it certain.
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case):
    if case["levels"] != TAUS:
        raise ValueError("levels are not the five scored ones")
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    target_hour = max(0.0, parse_time(case["target"]) / 3600.0)
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)
    hours = [max(0.0, parse_time(h["at"]) / 3600.0) for h in history]
    tracked = [h.get("players_tracked") for h in history]
    n = len(case["trophies"])
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rates = []
        for h in history:
            rs = h.get("rates") or []
            rates.append(rs[i] if i < len(rs) else None)
        rows.append({"id": trophy["id"], "q": forecast_trophy(rates, tracked, hours, target_hour, horizon_days)})
    return {"trophies": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        (OUT_DIR / (path.stem + ".out")).write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
