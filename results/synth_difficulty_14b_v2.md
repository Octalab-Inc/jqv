### Qwen/Qwen3-14B (slot, split=test_14b-hardfam-v2)

- long_policy: no results (synth-long_policy-test_slot_qwen3-14b_14b-hardfam-v2.npz missing)
**temporal_numeric**: n=500, accuracy 0.554

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| business_day_deadline | 99 | 0.606 | 0.11 |
| dst_cutoff | 58 | 0.328 | 0.00 |
| prorated_invoice | 70 | 0.400 | 0.00 |
| service_months_band | 93 | 0.677 | 0.00 |
| unit_threshold | 70 | 0.529 | 0.00 |
| warranty_month_end_tz | 110 | 0.636 | 0.05 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 1 | 0.000 | - |
| 3 | 210 | 0.543 | 0.07 |
| 4 | 242 | 0.558 | 0.00 |
| 5 | 44 | 0.614 | 0.12 |
| 6 | 3 | 0.333 | 0.00 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 229 | 0.498 | 0.08 |
| noul | 182 | 0.670 | 0.00 |
| score | 89 | 0.461 | 0.00 |

- probability: no results (synth-probability-test_slot_qwen3-14b_14b-hardfam-v2.npz missing)
**temporal_v2**: n=500, accuracy 0.614

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| deadline_boolean | 102 | 0.618 | 0.11 |
| effective_expiry | 103 | 0.718 | 0.08 |
| fx_lines_cap | 99 | 0.475 | 0.16 |
| multi_condition | 102 | 0.578 | 0.28 |
| term_vs_cap | 94 | 0.681 | 0.20 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 4 | 100 | 0.630 | 0.09 |
| 5 | 217 | 0.645 | 0.12 |
| 6 | 96 | 0.438 | 0.36 |
| 7 | 52 | 0.750 | 0.13 |
| 8 | 35 | 0.657 | 0.17 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 315 | 0.530 | 0.14 |
| noul | 154 | 0.844 | 0.15 |
| score | 31 | 0.323 | 0.43 |

**all families by dependency_hops**

| hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 1 | 0.000 | - |
| 3 | 210 | 0.543 | 0.07 |
| 4 | 342 | 0.579 | 0.04 |
| 5 | 261 | 0.640 | 0.12 |
| 6 | 99 | 0.434 | 0.35 |
| 7 | 52 | 0.750 | 0.13 |
| 8 | 35 | 0.657 | 0.17 |

