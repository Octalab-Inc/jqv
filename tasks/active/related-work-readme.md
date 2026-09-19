---
title: 関連プロジェクト（Simple Jev, NanoJev, SemIf, openjev-sglang, Hydragen, DeFT）との位置づけを README に書く
status: pending
priority: P3
created_at: 2026-09-20T05:23:55+09:00
depends_on: []
---

# Goal

jqv と同じ仮説（生成しない直接 readout、shared state、学習 head、Brier 学習、shared-prefix attention）に到達している公開プロジェクトとの
位置づけを README に 1 節で示し、jqv の独自性（engine の分解と数値等価性、isolation の負対照、一般 benchmark での校正と温度転移）を明確にする。

# Context

- ユーザー提供の調査メモ: Simple Jev（Featherless、next-token logits + shared KV + RFDT/LoRA）、NanoJev（Qwen3-0.6B、専用 head、CE/Brier、
  RLCD 風、simulator 環境）、SemIf（旧 OpenJev、Qwen3.5-4B、direct logits、自作 benchmark 81.3%、TypeSafe subset agreement 84.5%、
  direct 1.02 s vs JSON 生成 5.33 s）、openjev-sglang（Qwen3.6-35B-A3B + SGLang）、Hydragen、DeFT。
- **これらは未検証。** README に書く前に URL を確認し、確認できないものは「未確認」と明記するか載せない。
- jqv 側の対応する結果は README の各節にある（engine 等価性、isolation、温度転移、λ sweep、D3）。

# Scope

## In

- 各プロジェクトの URL と要点を確認（WebSearch / WebFetch）。数値は一次情報から引く
- README に「関連プロジェクト」節: 表（project / 近さ / 特徴 / jqv との差）と、jqv の独自性 4 点
- NanoJev の評価方針（ECE だけでなく NLL/Brier、risk-coverage、OOD を分ける）が jqv の `compare_runs.py` / 転移表と対応することを書く

## Out

- コードの変更、他プロジェクトの実行

# Success

- README に関連プロジェクト節があり、載せた各項目に URL がある。確認できなかった項目は「未確認」と明記されている
- jqv の独自性が、他プロジェクトが「明記していること」（校正していない等）との対比で書かれている

# Verify

README を読んで URL が開くことを確認する。
