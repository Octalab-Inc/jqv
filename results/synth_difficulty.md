# Synthetic hard-family data: difficulty check against JevBench

Generated with `python -m jqv.synth.generate --family all --n-train 2000 --n-dev 300 --n-test 500 --seed 0`
(generator version 0.1; temporal_numeric and probability hardened after a first round of evaluation, see the history below). Every answer comes from a solver; `dependency_hops` = derived facts in the solver's trace,
`reasoning_depth` = longest derivation chain. Contamination check against the 231 JevBench public items:
0 shared word 8-grams, no reused IDs or invented names (`scripts/synth_contamination.py`).

Evaluation: `scripts/eval.py --dataset synth:<family>:dev --n 0 --n-val 0 --engine packed` (zero-shot, raw
probabilities; accuracy does not depend on the temperature), tables by `scripts/synth_difficulty.py`.

## JevBench public hard, per family (`scripts/jevbench_families.py`, served-T runs)

| run | long_policy | temporal_numeric | probability | multi_hop | all 111 |
|---|---:|---:|---:|---:|---:|
| qwen3-14b zero-shot | 5/19 = 0.26 | 5/15 = 0.33 | 5/10 = 0.50 | 10/18 | 61/111 = 0.550 |
| qwen3-32b zero-shot | 9/19 = 0.47 | 5/15 = 0.33 | 4/10 = 0.40 | 10/18 | 69/111 = 0.622 |

Target for the synthetic dev sets: 32B zero-shot within +-10 pt of the JevBench family accuracy
(long_policy 0.47, temporal_numeric 0.33, probability 0.40), with the caveat that the JevBench families have
10-19 items each (95% CI roughly +-20 pt).

## Synthetic dev (300 items per family)

### Qwen/Qwen3-14B (packed, split=dev)

**long_policy**: n=300, accuracy 0.333

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| equipment_breakdown | 51 | 0.412 | 0.25 |
| home_water | 82 | 0.256 | 0.15 |
| relocation_reimbursement | 59 | 0.085 | 0.58 |
| sla_credits | 49 | 0.673 | 1.00 |
| trip_cancellation | 59 | 0.339 | 0.44 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 3 | 101 | 0.238 | 0.45 |
| 4 | 131 | 0.405 | 0.54 |
| 5 | 38 | 0.105 | 0.15 |
| 6 | 25 | 0.720 | 0.11 |
| 7 | 5 | 0.200 | 1.00 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 214 | 0.248 | 0.38 |
| noul | 49 | 0.592 | 0.40 |
| score | 37 | 0.486 | 1.00 |

**temporal_numeric**: n=300, accuracy 0.270

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| business_day_deadline | 54 | 0.111 | 0.90 |
| dst_cutoff | 49 | 0.429 | 0.83 |
| prorated_invoice | 39 | 0.154 | 0.31 |
| service_months_band | 52 | 0.500 | 0.92 |
| unit_threshold | 28 | 0.107 | 0.86 |
| warranty_month_end_tz | 78 | 0.244 | 0.97 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 1 | 0.000 | 1.00 |
| 3 | 131 | 0.237 | 0.96 |
| 4 | 140 | 0.321 | 0.64 |
| 5 | 27 | 0.185 | 0.64 |
| 6 | 1 | 0.000 | 1.00 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 152 | 0.132 | 0.67 |
| noul | 106 | 0.500 | 0.84 |
| score | 42 | 0.190 | 0.94 |

**probability**: n=300, accuracy 0.447

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| acceptance_sampling | 57 | 0.421 | 0.94 |
| draw_outcomes | 15 | 0.600 | - |
| ev_choice | 46 | 0.609 | 0.61 |
| redundancy | 47 | 0.234 | 0.61 |
| screening_posterior | 87 | 0.356 | 0.29 |
| supplier_mix | 48 | 0.646 | 0.89 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 47 | 0.234 | 0.61 |
| 3 | 60 | 0.433 | 0.91 |
| 4 | 135 | 0.393 | 0.45 |
| 5 | 58 | 0.759 | 0.42 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 214 | 0.383 | 0.58 |
| noul | 86 | 0.605 | 0.52 |

**all families by dependency_hops**

| hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 48 | 0.229 | 0.63 |
| 3 | 292 | 0.277 | 0.74 |
| 4 | 406 | 0.372 | 0.55 |
| 5 | 123 | 0.431 | 0.38 |
| 6 | 26 | 0.692 | 0.16 |
| 7 | 5 | 0.200 | 1.00 |

### Qwen/Qwen3-32B (packed, split=dev)

**long_policy**: n=300, accuracy 0.383

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| equipment_breakdown | 51 | 0.373 | 0.25 |
| home_water | 82 | 0.439 | 0.20 |
| relocation_reimbursement | 59 | 0.102 | 0.77 |
| sla_credits | 49 | 0.673 | 1.00 |
| trip_cancellation | 59 | 0.356 | 0.37 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 3 | 101 | 0.228 | 0.53 |
| 4 | 131 | 0.420 | 0.57 |
| 5 | 38 | 0.395 | 0.15 |
| 6 | 25 | 0.760 | 0.17 |
| 7 | 5 | 0.600 | 1.00 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 214 | 0.285 | 0.44 |
| noul | 49 | 0.592 | 0.45 |
| score | 37 | 0.676 | 0.92 |

**temporal_numeric**: n=300, accuracy 0.323

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| business_day_deadline | 54 | 0.093 | 0.90 |
| dst_cutoff | 49 | 0.551 | 1.00 |
| prorated_invoice | 39 | 0.051 | 0.77 |
| service_months_band | 52 | 0.481 | 0.96 |
| unit_threshold | 28 | 0.143 | 0.86 |
| warranty_month_end_tz | 78 | 0.436 | 0.62 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 1 | 0.000 | 1.00 |
| 3 | 131 | 0.313 | 0.91 |
| 4 | 140 | 0.336 | 0.78 |
| 5 | 27 | 0.333 | 0.73 |
| 6 | 1 | 0.000 | 1.00 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 152 | 0.224 | 0.78 |
| noul | 106 | 0.519 | 0.83 |
| score | 42 | 0.190 | 0.97 |

**probability**: n=300, accuracy 0.453

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| acceptance_sampling | 57 | 0.439 | 0.90 |
| draw_outcomes | 15 | 0.800 | - |
| ev_choice | 46 | 0.435 | 0.56 |
| redundancy | 47 | 0.255 | 0.72 |
| screening_posterior | 87 | 0.402 | 0.32 |
| supplier_mix | 48 | 0.667 | 0.74 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 47 | 0.255 | 0.72 |
| 3 | 60 | 0.400 | 0.88 |
| 4 | 135 | 0.437 | 0.42 |
| 5 | 58 | 0.707 | 0.42 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 214 | 0.425 | 0.55 |
| noul | 86 | 0.523 | 0.56 |

**all families by dependency_hops**

| hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 48 | 0.250 | 0.74 |
| 3 | 292 | 0.301 | 0.75 |
| 4 | 406 | 0.397 | 0.60 |
| 5 | 123 | 0.528 | 0.39 |
| 6 | 26 | 0.731 | 0.21 |
| 7 | 5 | 0.600 | 1.00 |


## Summary against the targets

| family | 14B dev | 32B dev | JevBench 32B (target ±10 pt) | verdict |
|---|---:|---:|---:|---|
| long_policy | 0.333 | 0.383 | 0.47 (9/19) | within range |
| temporal_numeric | 0.270 | 0.323 | 0.33 (5/15) | within range |
| probability | 0.447 | 0.453 | 0.40 (4/10) | within range |

History: the first generation gave 14B dev accuracies of temporal_numeric 0.507 and probability 0.657, above the
32B targets, so both generators were hardened: the distractor note must disagree with the truth in most items
(temporal 60%, probability 54% of dev items), the share of two-option `noul` questions was reduced in favour of
4-6-option `choice` questions, and the probability scenarios are redrawn within the same scenario when a draw is
rejected (otherwise scenarios with high rejection rates were under-represented). After the first hardening the 32B
scored temporal 0.323 and probability 0.550; a second round on probability only (fewer `noul`, wider parameter spaces
for the small scenarios) brought it to 0.453.

Observations: the 14B/32B follow the distractor note in most items where the note disagrees with the truth
(surface-answer rate 0.6-1.0 in temporal_numeric), which is the same failure mode the JevBench hard items are built
around. Accuracy is not monotone in `dependency_hops`: in long_policy the 6-hop items (the "pay within the sublimit"
decisions) are the easiest, so hops measure the length of the derivation, not the difficulty by itself.
