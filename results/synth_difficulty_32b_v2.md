### Qwen/Qwen3-32B (slot, split=test_32b-hardfam-v2)

**long_policy**: n=500, accuracy 0.564

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| equipment_breakdown | 97 | 0.691 | 0.10 |
| home_water | 132 | 0.545 | 0.18 |
| relocation_reimbursement | 93 | 0.538 | 0.06 |
| sla_credits | 83 | 0.602 | 0.74 |
| trip_cancellation | 95 | 0.453 | 0.32 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 3 | 168 | 0.726 | 0.12 |
| 4 | 224 | 0.513 | 0.36 |
| 5 | 74 | 0.405 | 0.19 |
| 6 | 30 | 0.400 | 0.19 |
| 7 | 4 | 0.750 | 0.00 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 339 | 0.510 | 0.16 |
| noul | 100 | 0.750 | 0.20 |
| score | 61 | 0.557 | 0.73 |

**temporal_numeric**: n=500, accuracy 0.624

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| business_day_deadline | 99 | 0.636 | 0.10 |
| dst_cutoff | 58 | 0.448 | 0.40 |
| prorated_invoice | 70 | 0.529 | 0.00 |
| service_months_band | 93 | 0.763 | 0.02 |
| unit_threshold | 70 | 0.571 | 0.04 |
| warranty_month_end_tz | 110 | 0.682 | 0.00 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 1 | 1.000 | - |
| 3 | 210 | 0.629 | 0.09 |
| 4 | 242 | 0.628 | 0.02 |
| 5 | 44 | 0.568 | 0.00 |
| 6 | 3 | 0.667 | 0.00 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 229 | 0.607 | 0.04 |
| noul | 182 | 0.665 | 0.10 |
| score | 89 | 0.584 | 0.01 |

**probability**: n=500, accuracy 0.770

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| acceptance_sampling | 116 | 0.733 | 0.25 |
| draw_outcomes | 25 | 0.760 | 1.00 |
| ev_choice | 78 | 0.692 | 0.05 |
| redundancy | 67 | 0.627 | 0.04 |
| screening_posterior | 145 | 0.841 | 0.01 |
| supplier_mix | 69 | 0.913 | 0.06 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 67 | 0.627 | 0.04 |
| 3 | 107 | 0.729 | 0.24 |
| 4 | 221 | 0.769 | 0.03 |
| 5 | 105 | 0.905 | 0.03 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 326 | 0.706 | 0.10 |
| noul | 174 | 0.891 | 0.04 |

**temporal_v2**: n=500, accuracy 0.716

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| deadline_boolean | 102 | 0.716 | 0.06 |
| effective_expiry | 103 | 0.806 | 0.06 |
| fx_lines_cap | 99 | 0.515 | 0.10 |
| multi_condition | 102 | 0.775 | 0.14 |
| term_vs_cap | 94 | 0.766 | 0.01 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 4 | 100 | 0.660 | 0.07 |
| 5 | 217 | 0.793 | 0.04 |
| 6 | 96 | 0.552 | 0.23 |
| 7 | 52 | 0.788 | 0.00 |
| 8 | 35 | 0.743 | 0.03 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 315 | 0.648 | 0.06 |
| noul | 154 | 0.935 | 0.05 |
| score | 31 | 0.323 | 0.40 |

**all families by dependency_hops**

| hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 68 | 0.632 | 0.04 |
| 3 | 485 | 0.685 | 0.13 |
| 4 | 787 | 0.639 | 0.11 |
| 5 | 440 | 0.732 | 0.06 |
| 6 | 129 | 0.519 | 0.22 |
| 7 | 56 | 0.786 | 0.00 |
| 8 | 35 | 0.743 | 0.03 |

