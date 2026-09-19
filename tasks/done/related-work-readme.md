---
title: 関連プロジェクト（Simple Jev, NanoJev, SemIf, openjev-sglang, Hydragen, DeFT）との位置づけを README に書く
status: done
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

# Result

## Changed

- `README.md`: 「関連プロジェクト」節（表 8 件 + 一覧 3 件、URL 付き。jqv の独自性 4 点、NanoJev の評価方針との対応、RLCD 型を次候補とする整理）

## Verified

- 各 URL は WebSearch で 2026-09-20 に実在を確認（simple-jev, NanoJev, SemIf, openjev-sglang, mini-jev, jev-forge, awesome-jev, LightJev, qwen-rlcd, Hydragen arXiv:2402.05099, DeFT arXiv:2404.00242）
- 数値（SemIf の 81.3% / 84.5% / 1.02 s vs 5.33 s、Hydragen の 32x、DeFT の 73〜99%）は各 README / abstract の自己申告値であることを明記

## Deviations

- ユーザーの調査メモにあった数値のうち、一次情報で確認できなかったものは README に載せず「自己申告」の注記で扱った

## Remaining

- なし
