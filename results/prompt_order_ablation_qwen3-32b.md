### Prompt order ablation (qwen3-32b, zero-shot packed, served T fitted per layout on MMLU val 400)

| layout | shared prefix | easy | standard | hard | hard vs A (won/lost, p) | hard ECE | hard Brier | hard ordinal MAE | standard ECE |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A state→Q | yes | 1.000 | 0.944 (68/72) | **0.613** (68/111) | - | 0.127 | 0.516 | 0.766 | 0.101 |
| B state→Q→Q | yes | 1.000 | 0.889 (64/72) | **0.622** (69/111) | 2/1, p=1.000 | 0.090 | 0.509 | 0.778 | 0.054 |

| layout | adversarial | ambiguous | judge_hard | long_policy | multi_hop | probability | routing_hard | temporal_numeric | tradeoff | trap | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A state→Q | 6/6 | 5/7 | 14/17 | 9/19 | 10/18 | 4/10 | 5/5 | 4/15 | 3/6 | 8/8 | 68/111 |
| B state→Q→Q | 6/6 | 5/7 | 14/17 | 9/19 | 10/18 | 5/10 | 5/5 | 5/15 | 2/6 | 8/8 | 69/111 |

| layout | MMLU acc (800) | T (MMLU val) | NLL raw / +T | ECE raw / +T | q/s (packed, 1 q/state) |
|---|---:|---:|---:|---:|---:|
| A state→Q | 0.806 | 3.023 | 0.946 / 0.535 | 0.139 / 0.025 | 5.646 |
| B state→Q→Q | 0.825 | 2.961 | 0.883 / 0.494 | 0.127 / 0.035 | 2.523 |
| C Q→state→Q | 0.825 | 2.961 | 0.885 / 0.495 | 0.127 / 0.033 | 2.270 |
| D Q→state | 0.810 | 3.014 | 0.950 / 0.536 | 0.141 / 0.030 | 2.704 |
