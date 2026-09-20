| backbone | params | B zero-shot: MMLU / JMMLU | B ECE raw / +T (T) | B + perm_avg: MMLU / JMMLU | perm_avg ECE raw / +T | 5-shot + perm_avg: MMLU | slot+LoRA: MMLU / JMMLU | slot ECE raw / +T | packed q/s (S=2k, Q=100) plain / perm_avg |
|---|---:|---:|---|---:|---|---:|---:|---|---:|
| qwen3-1.7b | 1.7B | 0.554 / 0.466 | 0.413 / 0.080 (12.0) | 0.584 / 0.485 | 0.195 / 0.063 | 0.578 | 0.575 / 0.505 (slot_lora) | 0.137 / 0.044 | 48.6 / 28.1 |
| qwen3-14b | 14.8B | 0.750 / 0.710 | 0.207 / 0.042 (5.1) | 0.781 / 0.729 | 0.113 / 0.047 | 0.782 | 0.757 / 0.711 (qwen3-14b_slot_brier0) | 0.114 / 0.045 | 9.3 / 4.4 |
| Jev (TypeSafe, Hume 2025) | ? | **0.918** / - | 0.031 (zero-shot, no T) | - | - | - | - | - | 30k tok in ~160 ms |
