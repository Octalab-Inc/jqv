---
title: GB10（DGX Spark）2 台で jqv の学習・評価を回せるようにする（鍵認証、環境構築、runbook）
status: pending
priority: P1
created_at: 2026-09-22T12:23:12+09:00
depends_on: []
---

# Goal

Tailscale 上の GB10 2 台（promaxgb10-13fd = gb10a、promaxgb10-1452 = gb10b。NVIDIA GB10、CUDA 13.0、統合メモリ 121 GB、aarch64 Ubuntu 24.04）で
jqv の学習（`scripts/train_head.py`）と評価（`scripts/eval.py`、JevBench ハーネス）を Mac から鍵認証で起動・監視・回収できる状態にする。
最初の実ジョブは 32B targeted LoRA（`hard-family-targeted-lora` の続き）。

# Context

- Mac（MPS）では 32B の 1 step が 88 秒で、600 step に 14 時間かかるためユーザーが GB10 への移行を指示した。GB10 側なら時間がかかってもよい（DDP は今回作らない）。
- 両機とも uv・torch なし、docker・git・tmux あり、HF / GitHub に到達可。DeepSeek-V4-Flash の vLLM コンテナ（`deepseek-v4-flash-vllm-dspark-1`、2 台で TP=2、GPU メモリ 103 GB）が稼働中で、ユーザーの指示により停止してよい。
- ログインは `.env` の USER / PW（git 未追跡、`.gitignore` 済み）。パスワードは鍵登録に 1 回だけ使い、表示・記録しない。
- 最大の不確定要素は aarch64 + CUDA 13（sm_121）向けの torch wheel。PyPI → cu130 index → NGC コンテナの順で試す。
- 計画: `~/.claude/plans/qwen-jev-pfn-insight-scan-https-www-pref-functional-gizmo.md`。

# Scope

## In

- Mac: `~/.ssh/config` の `gb10a` / `gb10b`、公開鍵登録、`task/hard-family-targeted-lora` の push。
- 両機: vLLM コンテナ停止（再開手順を記録）、uv、repo clone、`uv sync`、torch の CUDA 動作確認と必要な index の記録、Qwen3 1.7B / 14B / 32B の取得、synth train の再生成、モデル不要テストと CUDA での engine 等価性テスト、14B smoke（4 step）で step 時間とメモリの実測。
- `docs/gb10.md`（runbook: ホスト、鍵、tmux、環境構築、起動・監視・回収、vLLM の停止と再開）。
- 32B run の起動（A）と zero-shot 基準測定（B）は本タスクの成果物として起動まで、結果の記録は `hard-family-targeted-lora`。

## Out

- 2 ノード DDP（計画に設計メモのみ）。
- vLLM の再開（ユーザーまたは所有者が判断）。
- JevBench 公開 endpoint の GB10 への移設。

# Success

- `ssh gb10a hostname` / `ssh gb10b hostname` が鍵認証で通る。
- 両機で `uv run pytest tests/test_engines_equivalence.py tests/test_isolation.py` が CUDA で通る。
- 14B smoke の step 時間が Mac（47 秒/step）より速い値で記録されている。
- 32B の学習が A の tmux で動き、B で 32B zero-shot の基準（synth test 3 family、MMLU、JMMLU）が取れている。
- `docs/gb10.md` にパスワードなしで再現できる手順がある。

# Verify

```bash
ssh -o BatchMode=yes gb10a 'nvidia-smi --query-gpu=name,memory.used --format=csv; free -g | head -2'
ssh gb10a 'cd ~/jqv && uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"'
ssh gb10a 'cd ~/jqv && uv run pytest tests/test_engines_equivalence.py tests/test_isolation.py -q'
ssh gb10a 'tmux capture-pane -pt train32 | tail -3'
```

# Open Questions

- なし。
