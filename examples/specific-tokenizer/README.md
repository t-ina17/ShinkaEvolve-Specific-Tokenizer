# specific-tokenizer（ドメイン特化 形態素解析の探索）

この例は、ShinkaEvolveで「クエリと商品テキストのマッチ度」をfitnessとして、**ドメイン特化したトークナイザ/形態素解析**（主に`tokenize()`と`score_match()`）を探索するためのテンプレートです。

## 目的

- 入力: `query`（検索クエリ）と `product`（商品タイトル/説明など）
- 形態素解析器（SudachiPy）を使う場合は追加依存が必要です。未導入でもフォールバック（正規化＋分割＋文字n-gram）で動きます。導入: `uv pip install SudachiPy sudachidict_core`
- データに `label`（関連度）がある場合は、クエリごとにランキング評価（NDCG@10）をfitnessにします

## データ形式

このテンプレートは次の2種類を扱えます。

1) CSV（UTF-8）: 最低限 `query` と `product`（または `product_title`）が必要
2) Amazon ESCI-data（Shopping Queries Dataset）: repo root を `--data_path` に指定

推奨カラム:
- `query`: str
- `product`: str（または `product_title`）
- `label`: int（0/1や0..Nの関連度） ※任意
- `query_id`: str/int（同一クエリのグルーピング用） ※`label`を使う場合に推奨

サンプル: `data/sample.csv`

### Amazon ESCI-data（日本語のみで検証）

ECSI-dataをcloneしたディレクトリを `--data_path` に渡します。

※ Parquet読み込みのため `pyarrow` が必要です。未導入なら `uv pip install pyarrow`。

```bash
cd ShinkaEvolve/examples/specific-tokenizer
python evaluate.py \
	--program_path initial.py \
	--results_dir results_esci_jp \
	--data_path /Users/tina/work/study/sakana-ai/prod/data/esci-data \
	--dataset esci \
	--esci_locale jp \
	--esci_split train \
	--esci_version small \
	--max_queries 200 \
	--max_products_per_query 40 \
	--product_text_fields title_brand
```

## まず動かす（評価のみ）

```bash
cd ShinkaEvolve/examples/specific-tokenizer
python evaluate.py --program_path initial.py --results_dir results --data_path data/sample.csv
```

## 進化を回す

```bash
cd ShinkaEvolve/examples/specific-tokenizer
python run_evo.py
```

- `initial.py` の `EVOLVE-BLOCK-START/END` 内が主に進化対象です
- 本番データ（Amazon ECSI-data など）を使う場合は `--data_path` を差し替え、`run_evo.py`の`TASK_DATA_PATH`も更新してください

## 可視化（精度推移・探索木）

`circle_packing` と同様に、SQLiteの進化DBから **スコア推移** と **探索木（lineage tree）** を表示できます。

### WebUI（推奨）

結果ディレクトリを指定して起動します（DBは自動で見つけます）。

```bash
cd ShinkaEvolve/examples/specific-tokenizer
shinka_visualize results_specific_tokenizer_oss120b_long --port 8888 --open
```

DBを直接指定したい場合:

```bash
cd ShinkaEvolve/examples/specific-tokenizer
shinka_visualize --db results_specific_tokenizer_oss120b_long/evolution_db_oss120b_long.sqlite --port 8888 --open
```

### 画像出力（PNG）

同梱のスクリプトで、改善曲線と探索木をPNGに保存できます。

```bash
cd ShinkaEvolve/examples/specific-tokenizer
python viz.py --results_dir results_specific_tokenizer_oss120b_long --title "ECSI JP tokenizer"
```

出力先: `results_specific_tokenizer_oss120b_long/viz/`

## 使い分けの目安

- ラベルあり（推薦）: `label` と `query_id` を用意 → NDCG@10 をfitness
- ラベルなし: 与えられたペアの平均マッチスコア（弱い教師）

## 注意

- 大規模データは評価が重くなるので、まずは`--max_rows`でサブサンプルしてから回すのがおすすめです
- 形態素解析器（SudachiPy等）を使う場合は、追加依存関係が必要です（このテンプレートは外部辞書無しでも動くようにフォールバック実装にしています）
