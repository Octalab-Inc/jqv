### Qwen/Qwen3-32B (slot, split=test_32b-hardfam-gb10)

**long_policy**: n=500, accuracy 0.596

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| equipment_breakdown | 97 | 0.732 | 0.16 |
| home_water | 132 | 0.652 | 0.18 |
| relocation_reimbursement | 93 | 0.505 | 0.29 |
| sla_credits | 83 | 0.614 | 0.86 |
| trip_cancellation | 95 | 0.453 | 0.38 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 3 | 168 | 0.637 | 0.26 |
| 4 | 224 | 0.585 | 0.43 |
| 5 | 74 | 0.635 | 0.19 |
| 6 | 30 | 0.333 | 0.23 |
| 7 | 4 | 0.750 | 0.00 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 339 | 0.555 | 0.25 |
| noul | 100 | 0.740 | 0.34 |
| score | 61 | 0.590 | 0.70 |

**temporal_numeric**: n=500, accuracy 0.668

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| business_day_deadline | 99 | 0.737 | 0.06 |
| dst_cutoff | 58 | 0.552 | 0.70 |
| prorated_invoice | 70 | 0.514 | 0.00 |
| service_months_band | 93 | 0.817 | 0.02 |
| unit_threshold | 70 | 0.643 | 0.11 |
| warranty_month_end_tz | 110 | 0.655 | 0.16 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 1 | 1.000 | - |
| 3 | 210 | 0.690 | 0.13 |
| 4 | 242 | 0.649 | 0.03 |
| 5 | 44 | 0.659 | 0.18 |
| 6 | 3 | 0.667 | 0.00 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 229 | 0.633 | 0.05 |
| noul | 182 | 0.703 | 0.18 |
| score | 89 | 0.685 | 0.01 |

**probability**: n=500, accuracy 0.766

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| acceptance_sampling | 116 | 0.724 | 0.29 |
| draw_outcomes | 25 | 0.800 | 1.00 |
| ev_choice | 78 | 0.692 | 0.05 |
| redundancy | 67 | 0.642 | 0.04 |
| screening_posterior | 145 | 0.834 | 0.00 |
| supplier_mix | 69 | 0.884 | 0.12 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 67 | 0.642 | 0.04 |
| 3 | 107 | 0.729 | 0.24 |
| 4 | 221 | 0.751 | 0.07 |
| 5 | 105 | 0.914 | 0.00 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 326 | 0.702 | 0.12 |
| noul | 174 | 0.885 | 0.04 |

**all families by dependency_hops**

| hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 68 | 0.647 | 0.04 |
| 3 | 485 | 0.680 | 0.20 |
| 4 | 687 | 0.661 | 0.16 |
| 5 | 223 | 0.771 | 0.10 |
| 6 | 33 | 0.364 | 0.21 |
| 7 | 4 | 0.750 | 0.00 |

