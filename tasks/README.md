# Tasks

このディレクトリは、コーディングエージェントと人間が共有するタスクキューです。
このファイルはこのリポジトリでの規約を定めます。
キュー操作のコマンド (`task_queue.py`) は task-builder スキルが提供し、実行方法はスキルの SKILL.md にあります。

## 識別子

タスクの識別子はファイル名(slug)そのものです。
`active/search-timeout.md` なら `search-timeout` で、フロントマターに `id` は持ちません。
英小文字、数字、ハイフンで2〜48文字。内容が分かる名前を付けます。

採番機を持たないので、別ブランチで同時にタスクを作っても番号が衝突しません。
同じslugを2人が作った場合は、マージ時にgitが衝突として止めます。

## Status

- `draft`: 要件整理中。実装しない
- `pending`: 実行可能で未着手
- `blocked`: 依存関係または外部判断待ち
- `done`: SuccessとVerifyを完了
- `cancelled`: 実行しない

着手中を表す値はありません。着手状態は作業ブランチ `task/<slug>` が持ちます。

## Priority

- `P0`: 緊急。本番障害、重大なセキュリティ問題、全体ブロッカー
- `P1`: 重要。次回リリース必須、主要機能障害、複数タスクのブロッカー
- `P2`: 通常の計画タスク
- `P3`: 後回しにできる改善または調査

## Task creation

タスクファイルは task-builder スキルの `task_queue.py new` で作ります。
`created_at` が自動で入り、ファイル名がそのまま識別子になります。

## Task selection

次の条件を満たすタスクから1件を選びます。

1. `status` が `pending`
2. `depends_on` の全タスクが `done`(未マージで参照できないものは未達として扱う)
3. 作業ブランチがない(着手中でない)
4. 最も高いpriority
5. 同じpriorityなら `created_at` が古い順、同時刻ならslug順

この規則で1件選ぶのが `task_queue.py next`、処理順に全件並べるのが `next --all` です。

## Claim

着手は `task_queue.py claim` で行います。
作業ブランチ `task/<slug>` を作ってpushし、pushできた側が所有者です。
着手中の一覧は `claims`、着手をやめるときは `release <slug>` です。

同時に取りにいくと、後発のpushが弾かれて次の候補へ移ります。
mainへのpushは不要です。タスクファイルの `status` は着手しても変えません。

放置された作業ブランチは `claims` がSTALEとして表示します。自動では消しません。

## Completion

以下を満たす場合のみ `done` にします。

- Successを満たす
- Verifyを実行する
- 実行できなかった確認を明記する
- 仕様からの逸脱を記録する
- 残課題を記録する

完了時はタスクファイルを `done/` へ移動し、実装と同じPRに含めます。
PRがマージされて作業ブランチが消えると、着手状態も解けます。

## Rules

- フロントマターに `id` を書かない
- claimせずに着手しない。statusを書き換えて着手状態を表さない
- 他エージェントの作業ブランチを、放棄を確認せずに消さない
- `draft` のタスクを実装しない
- Goal、Scope、Successを勝手に変更しない
- Scope外の変更が必要な場合は別タスクを提案する
- 未決事項を暗黙に決定しない
- 実行していないテストを成功扱いしない
- タスク本文に長い進捗ログを書かない

## Layout

- `active/`: 未完了タスク (`draft` / `pending` / `blocked`)
- `done/`: 完了・中止タスク (`done` / `cancelled`)。完了時にファイルを移動する
- `_template.md`: 新規タスクのテンプレート

`active/` と `done/` には `.gitkeep` を置きます。
空ディレクトリはgitに載らず、clone先やworktreeで存在しなくなるためです。
