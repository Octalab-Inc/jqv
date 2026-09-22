### Qwen/Qwen3-32B (packed, split=test)

**long_policy**: n=500, accuracy 0.380

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| equipment_breakdown | 97 | 0.381 | 0.31 |
| home_water | 132 | 0.364 | 0.24 |
| relocation_reimbursement | 93 | 0.226 | 0.84 |
| sla_credits | 83 | 0.578 | 1.00 |
| trip_cancellation | 95 | 0.379 | 0.47 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 3 | 168 | 0.244 | 0.61 |
| 4 | 224 | 0.455 | 0.59 |
| 5 | 74 | 0.338 | 0.27 |
| 6 | 30 | 0.600 | 0.12 |
| 7 | 4 | 1.000 | 0.00 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 339 | 0.280 | 0.47 |
| noul | 100 | 0.640 | 0.42 |
| score | 61 | 0.508 | 0.91 |

**temporal_numeric**: n=500, accuracy 0.302

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| business_day_deadline | 99 | 0.162 | 0.93 |
| dst_cutoff | 58 | 0.500 | 1.00 |
| prorated_invoice | 70 | 0.014 | 0.89 |
| service_months_band | 93 | 0.409 | 1.00 |
| unit_threshold | 70 | 0.343 | 0.83 |
| warranty_month_end_tz | 110 | 0.391 | 0.77 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 1 | 1.000 | - |
| 3 | 210 | 0.243 | 0.94 |
| 4 | 242 | 0.339 | 0.86 |
| 5 | 44 | 0.364 | 0.88 |
| 6 | 3 | 0.333 | 0.50 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 229 | 0.175 | 0.87 |
| noul | 182 | 0.511 | 0.86 |
| score | 89 | 0.202 | 0.99 |

**probability**: n=500, accuracy 0.468

| scenario | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| acceptance_sampling | 116 | 0.353 | 0.94 |
| draw_outcomes | 25 | 0.840 | 1.00 |
| ev_choice | 78 | 0.577 | 0.38 |
| redundancy | 67 | 0.299 | 0.61 |
| screening_posterior | 145 | 0.434 | 0.29 |
| supplier_mix | 69 | 0.638 | 0.71 |

| dependency_hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 67 | 0.299 | 0.61 |
| 3 | 107 | 0.336 | 0.89 |
| 4 | 221 | 0.443 | 0.41 |
| 5 | 105 | 0.762 | 0.36 |

| type | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| choice | 326 | 0.433 | 0.49 |
| noul | 174 | 0.534 | 0.56 |

**all families by dependency_hops**

| hops | n | accuracy | surface-answer rate |
|---|---:|---:|---:|
| 2 | 68 | 0.309 | 0.61 |
| 3 | 485 | 0.264 | 0.80 |
| 4 | 687 | 0.410 | 0.63 |
| 5 | 223 | 0.543 | 0.39 |
| 6 | 33 | 0.576 | 0.14 |
| 7 | 4 | 1.000 | 0.00 |

