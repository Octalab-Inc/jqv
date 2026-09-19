| backbone | params | B: MMLU acc | B: JMMLU acc | B: ECE raw / +T (T) | slot+LoRA run | slot: MMLU acc | slot: JMMLU acc | slot: ECE raw / +T | packed q/s (S=2k, Q=100) |
|---|---:|---:|---:|---|---|---:|---:|---|---:|
| qwen3-1.7b | 1.7B | 0.554 | 0.466 | 0.413 / 0.080 (12.0) | slot_lora | 0.575 | 0.505 | 0.137 / 0.044 | 48.6 |
| Jev (TypeSafe, Hume 2025) | ? | **0.918** | - | 0.031 (zero-shot, no T) | - | - | - | - | 30k tok in ~160 ms |
