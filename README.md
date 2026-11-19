# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/MYsTRONOUT/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/MYsTRONOUT/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase one week after launch, from the house's recorded reads of the trophy page. For each trophy it writes five quantiles (5, 25, 50, 75, 95) of the rate at the case's target.

How it forecasts: a trophy's rate moves on two forces — players earning it (earned counts rise as people play) and new players arriving (tracked counts rise, diluting every rate). The program fits earned counts and tracked counts apart in log space against ln(1 + hours since release), weighted toward recent reads, reads both at the target, and divides; a direct fit of the logit's own trend is blended in half-and-half. The spread is a random walk in logit units that grows with the horizon, plus the fits' residual noise, plus a floor for Exophase's refresh risk — its counts jump in bursts, and a jump dilutes every rate at once, so a long horizon gets a wide, honest band instead of a confident miss.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, no network. It reads every `<id>.in` case in the input directory and writes `<id>.out` (one row per trophy, five quantiles each) to the output directory. It runs deterministically: two runs write the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=86f2bh8y&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
