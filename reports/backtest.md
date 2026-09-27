# Walk-forward backtest - 2025-26 NBA season

All numbers are **out-of-sample**: every prediction is made by a model that never saw
that game, using features built only from games played before it.

## Summary (regular season, games 6-82 for each team)

| Protocol | Model | Games | Accuracy | Log loss | Brier |
|---|---|---|---|---|---|
| Rolling: train 5 games / test next 10 | decision tree | 1150 | 64.7% | 0.763 | 0.239 |
| Rolling: train 5 games / test next 10 | random forest | 1150 | 66.6% | 0.614 | 0.212 |
| Expanding: train all prior / test next 10 | decision tree | 1150 | 67.7% | 0.650 | 0.214 |
| Expanding: train all prior / test next 10 | random forest | 1150 | 68.1% | 0.604 | 0.208 |

Baselines on the same games: home team wins 55.0%, better record wins 67.7%, better last-10 net rating wins 65.9%.

## Playoff holdout (trained on the full regular season)

| Model | Games | Accuracy | Log loss |
|---|---|---|---|
| decision tree | 91 | 59.3% | 0.655 |
| random forest | 91 | 61.5% | 0.649 |

Playoff baselines: home team 56.0%, better record 61.5%.

## Fold detail (rolling 5 / 10)

| Train games | Test games | Test n | Tree acc. | Forest acc. | Better-record acc. |
|---|---|---|---|---|---|
| 1-5 | 6-15 | 150 | 73.3% | 65.3% | 69.3% |
| 11-15 | 16-25 | 150 | 59.3% | 64.7% | 65.3% |
| 21-25 | 26-35 | 148 | 45.3% | 54.7% | 56.8% |
| 31-35 | 36-45 | 151 | 58.9% | 60.9% | 63.6% |
| 41-45 | 46-55 | 151 | 64.9% | 66.9% | 66.2% |
| 51-55 | 56-65 | 151 | 66.2% | 67.5% | 70.2% |
| 61-65 | 66-75 | 152 | 73.0% | 75.0% | 77.0% |
| 71-75 | 76-82 | 97 | 82.5% | 83.5% | 75.3% |

## Feature importance (random forest, full regular season)

- `NET_SZN_DIFF`: 0.351
- `LU_PER_DIFF`: 0.200
- `WINPCT_DIFF`: 0.173
- `NET_L10_DIFF`: 0.095
- `STAR_PER_DIFF`: 0.043
- `LU_USG_DIFF`: 0.040
- `CORE_PER_DIFF`: 0.031
- `LU_TS_DIFF`: 0.026
- `IS_HOME`: 0.017
- `MATCHUP_DIFF`: 0.015
- `CORE_N_DIFF`: 0.004
- `REST_DIFF`: 0.004
- `H2H_WINPCT`: 0.002
