### Prompt order ablation (qwen3-32b, zero-shot packed, served T fitted per layout on MMLU val 400)

| layout | shared prefix | easy | standard | hard | hard vs A (won/lost, p) | hard ECE | hard Brier | hard ordinal MAE | standard ECE |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A state→Q | yes | 1.000 | 0.944 (68/72) | **0.613** (68/111) | - | 0.127 | 0.516 | 0.766 | 0.101 |
| B state→Q→Q | yes | 1.000 | 0.889 (64/72) | **0.622** (69/111) | 2/1, p=1.000 | 0.090 | 0.509 | 0.778 | 0.054 |
| C Q→state→Q | no | 1.000 | 0.972 (70/72) | **0.640** (71/111) | 7/4, p=0.549 | 0.108 | 0.508 | 0.741 | 0.066 |
| D Q→state | no | 1.000 | 0.972 (70/72) | **0.577** (64/111) | 10/14, p=0.541 | 0.102 | 0.531 | 0.765 | 0.083 |

| layout | adversarial | ambiguous | judge_hard | long_policy | multi_hop | probability | routing_hard | temporal_numeric | tradeoff | trap | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A state→Q | 6/6 | 5/7 | 14/17 | 9/19 | 10/18 | 4/10 | 5/5 | 4/15 | 3/6 | 8/8 | 68/111 |
| B state→Q→Q | 6/6 | 5/7 | 14/17 | 9/19 | 10/18 | 5/10 | 5/5 | 5/15 | 2/6 | 8/8 | 69/111 |
| C Q→state→Q | 6/6 | 3/7 | 14/17 | 9/19 | 12/18 | 6/10 | 5/5 | 6/15 | 2/6 | 8/8 | 71/111 |
| D Q→state | 4/6 | 4/7 | 14/17 | 7/19 | 12/18 | 4/10 | 5/5 | 5/15 | 2/6 | 7/8 | 64/111 |

| layout | MMLU acc (800) | T (MMLU val) | NLL raw / +T | ECE raw / +T | q/s (packed, 1 q/state) |
|---|---:|---:|---:|---:|---:|
| A state→Q | 0.806 | 3.023 | 0.946 / 0.535 | 0.139 / 0.025 | 5.646 |
| B state→Q→Q | 0.825 | 2.961 | 0.883 / 0.494 | 0.127 / 0.035 | 2.523 |
| C Q→state→Q | 0.825 | 2.961 | 0.885 / 0.495 | 0.127 / 0.033 | 2.270 |
| D Q→state | 0.810 | 3.014 | 0.950 / 0.536 | 0.141 / 0.030 | 2.704 |
