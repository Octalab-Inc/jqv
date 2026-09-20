| backbone | params | B zero-shot: MMLU / JMMLU | B ECE raw / +T (T) | B + perm_avg: MMLU / JMMLU | perm_avg ECE raw / +T | 5-shot + perm_avg: MMLU | slot+LoRA: MMLU / JMMLU | slot ECE raw / +T | slot + perm_avg: MMLU | JevBench hard (public 111): B / perm_avg / slot | packed q/s (S=2k, Q=100) plain / perm_avg |
|---|---:|---:|---|---:|---|---:|---:|---|---:|---:|---:|
| qwen3-1.7b | 1.7B | 0.554 / 0.466 | 0.413 / 0.080 (12.0) | 0.584 / 0.485 | 0.195 / 0.063 | 0.578 | 0.575 / 0.505 (slot_lora) | 0.137 / 0.044 | - | 0.423 / 0.432 / 0.414 | 48.6 / 28.1 |
| qwen3-14b | 14.8B | 0.750 / 0.710 | 0.207 / 0.042 (5.1) | 0.781 / 0.729 | 0.113 / 0.047 | 0.782 | 0.757 / 0.711 (qwen3-14b_slot_brier0) | 0.114 / 0.045 | - | 0.550 / 0.568 / 0.586 | 9.3 / 4.4 |
| qwen3-32b | 32.8B | 0.809 / 0.771 | 0.137 / 0.023 (3.0) | 0.812 / 0.790 | 0.087 / 0.034 | - | 0.801 / 0.782 (qwen3-32b_slot_best) | 0.115 / 0.031 | 0.819 | 0.622 / 0.640 / 0.622 | 2.5 / - |
| Jev (TypeSafe, Hume 2025) | ? | **0.918** / - | 0.031 (zero-shot, no T) | - | - | - | - | - | - | **0.741** (534 決定、Benchmark Heaven) | 30k tok in ~160 ms |
