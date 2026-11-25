# npmi-hyperspherical-mf

[English](README.md) | 日本語

IEEE BigData 2026 採択論文 *Relational Learning between Stores and Products Free from the Effects of Product Popularity and Store Scale in Retail Chains* の公式実装リポジトリです．

- **著者**: Yoshimi Nakaya, Kei Yoshitake, Kai Yoshida, Momoka Yoshioka, Koei Yoshimura, Koyo Yamazaki, Shiko Kiyotake, Shota Masuyama, Ken Kobayashi, and Kazuhide Nakata (Institute of Science Tokyo)
- **発表会議**: IEEE International Conference on Big Data (IEEE BigData 2026)

---

## 概要

本リポジトリは，店舗・商品間の購買行列と補助情報から，商品人気や店舗規模の影響を緩和した店舗・商品埋め込みを学習する **超球面上の類似度学習フレームワーク** を提供します．

- **NPMI 変換 (前処理)**: 生の販売数の代わりに Normalized Pointwise Mutual Information (NPMI，値域 $`[-1, 1]`$) を教師データとして採用し，商品人気と店舗規模の影響を緩和します．
- **単位 $`\ell_2`$ ノルム制約**: 店舗・商品埋め込みベクトルに $`\lVert \mathbf{u}_i \rVert_2 = \lVert \mathbf{v}_j \rVert_2 = 1`$ を課し，内積をコサイン類似度 (値域 $`[-1, 1]`$) に一致させます．
- **補助情報制約**: 店舗特徴量 $`\mathbf{X}_U`$ と商品特徴量 $`\mathbf{X}_V`$ からエンコーダ $`f_\theta, g_\phi`$ を介して埋め込みを生成することで，購買履歴のない新規商品 (コールドスタート) にも対応します．

### 定式化

購買行列 $`\mathbf{R} \in \mathbb{Z}_{+}^{m \times n}`$，観測マスク $`\mathbf{M} \in \lbrace 0, 1 \rbrace^{m \times n}`$，店舗補助情報 $`\mathbf{X}_U \in \mathbb{R}^{m \times p}`$，商品補助情報 $`\mathbf{X}_V \in \mathbb{R}^{n \times q}`$ に対して，観測ペアの NPMI 行列 $`\mathbf{R}_{\text{NPMI}}`$ を以下の最適化問題により近似します．

$$
\min_{\theta, \phi} \; \left\lVert \mathbf{M} \odot \left( \mathbf{R}_{\text{NPMI}} - f_\theta(\mathbf{X}_U) g_\phi(\mathbf{X}_V)^\top \right) \right\rVert_F^2
$$

各エンコーダ $`f_\theta : \mathbb{R}^p \to \mathbb{R}^k`$，$`g_\phi : \mathbb{R}^q \to \mathbb{R}^k`$ は出力層に $`\ell_2`$ 正規化層を備え，任意の $`i \in \mathcal{I}, j \in \mathcal{J}`$ に対して $`\lVert f_\theta(\mathbf{x}^U_i) \rVert_2 = 1`$，$`\lVert g_\phi(\mathbf{x}^V_j) \rVert_2 = 1`$ を満たします．

---

## 環境構築

Python 3.11 以上および [uv](https://docs.astral.sh/uv/) の利用を推奨します．

```bash
git clone https://github.com/isct-nakatalab/npmi-hyperspherical-mf.git
cd npmi-hyperspherical-mf
uv sync
```

主な依存ライブラリ:
- PyTorch 2 (`torch`)
- NumPy, Pandas, SciPy, scikit-learn, PyYAML, PyArrow

---

## 使用方法

### 1. モデルの学習

YAML 設定ファイル，前処理済みデータディレクトリ，商品分割ファイル (`data_split.parquet`) を指定して学習を実行します．

```bash
# 提案手法 (Full) の学習
uv run python -m src.train_ablation \
  --config configs/m5/full.yaml \
  --seed 0 \
  --data-dir data/m5/processed \
  --split-parquet data/m5/data_split.parquet \
  --out outputs/m5/Full/seed_0
```

アブレーション条件 (`wo_npmi.yaml`, `wo_norm.yaml`, `wo_both.yaml`) も同じコマンドで `--config` を切り替えて実行できます．

ベースライン (`dropoutnet.yaml`, `content_concat.yaml`, `store_mean.yaml`) の学習は `src.train_baseline` を使用します．

```bash
# ベースラインの学習 (例: DropoutNet)
uv run python -m src.train_baseline \
  --config configs/m5/dropoutnet.yaml \
  --seed 0 \
  --data-dir data/m5/processed \
  --split-parquet data/m5/data_split.parquet \
  --out outputs/m5/DropoutNet/seed_0
```

### 2. 評価の実行

学習済みチェックポイント (`U.pt`, `V.pt` 等) に対して，ランキング指標 (Precision@$`K`$, Recall@$`K`$, MAP@$`K`$, NDCG@$`K`$, Personalization@$`K`$)，人気バイアス指標 (APLT, ARP, PS-Recall, head/tail-NDCG)，再構成指標 ($`R^2`$, $`\rho_{\text{store}}`$) を一括算出します．

```bash
uv run python -m src.eval_unified \
  --checkpoint outputs/m5/Full/seed_0 \
  --data-dir data/m5/processed \
  --split-parquet data/m5/data_split.parquet \
  --condition Full \
  --seed 0 \
  --out results/m5/Full/seed_0
```

### 3. 主な設定項目 (ハイパーパラメータ)

| パラメータ | 既定値 | 説明 |
| :--- | :--- | :--- |
| 店舗側エンコーダ $`f_\theta`$ | 線形写像 (バイアスなし) + $`\ell_2`$ 正規化 | 店舗補助情報 $`\mathbf{X}_U`$ を $`k`$ 次元へ射影 |
| 商品側エンコーダ $`g_\phi`$ | 2 層 MLP (隠れ次元 64) + $`\ell_2`$ 正規化 | 商品補助情報 $`\mathbf{X}_V`$ を $`k`$ 次元へ射影 |
| 埋め込み次元 $`k`$ | 32 | 店舗・商品埋め込みの次元数 |
| 最適化手法 | Adam (学習率 $`\eta = 10^{-3}`$，フルバッチ) | パラメータ $`\theta, \phi`$ の同時最適化 |
| 重み減衰 (weight decay) | $`10^{-4}`$ | `wo_norm`・`wo_both` 条件では 0 |
| 最大エポック数 / 早期終了 | $`10^5`$ / patience 500 | 検証 MSE を監視して最良重みを復元 |

---

## リポジトリ構成

```text
npmi-hyperspherical-mf/
├── configs/                # 実験設定ファイル (YAML)
├── data/                   # データ配置ディレクトリ (raw / processed は .gitignore 対象)
├── scripts/                # 前処理・学習・評価・集計スクリプト
├── src/                    # モデル・学習器・評価指標の実装
│   ├── mf/                 #   提案手法 (NPMISphereModel), アブレーションモデル, ベースライン
│   ├── train_ablation.py   #   提案手法およびアブレーション条件の学習スクリプト
│   ├── train_baseline.py   #   ベースライン (DropoutNet, ContentConcat, StoreMean) の学習スクリプト
│   └── eval_unified.py     #   統一評価スクリプト
├── tests/                  # 単体テスト
├── pyproject.toml          # 依存関係定義 (uv)
├── README.md               # 英語版 README
└── README_ja.md            # 本ファイル (日本語版 README)
```

---

## データセット

### M5 Walmart (公開データセット)

公開データセットである [M5 Forecasting - Accuracy (Kaggle)](https://www.kaggle.com/c/m5-forecasting-accuracy) を用いて動作確認および実験再現が可能です．

| 項目 | 値 |
| :--- | ---: |
| 店舗数 $`m`$ | 10 |
| 商品数 $`n`$ | 3,049 |
| 観測ペア数 $`\lvert \Omega \rvert`$ | 30,490 |
| 観測密度 $`\rho`$ | 1.000 |
| 店舗補助情報次元 $`p`$ | 13 (所在州 one-hot 3 次元 + 学習列の部門・カテゴリ別販売シェア 10 次元) |
| 商品補助情報次元 $`q`$ | 11 (カテゴリ・部門の one-hot ベクトル) |

元データを `data/m5/raw/` に配置して前処理スクリプトを実行することで，学習に必要な行列ファイル (`R`, `R_NPMI`, `mask`, `X_U`, `X_V`) と商品分割ファイル (`data_split.parquet`) を生成できます．独自の小売 POS データを用いる場合も，同一フォーマットの前処理済みファイルを準備することでそのまま学習・評価が可能です．

---

## 引用 (Citation)

本研究のコードやアイデアを利用される場合は，以下の論文を引用してください．

```bibtex
@inproceedings{nakaya2026relational,
  author    = {Nakaya, Yoshimi and Yoshitake, Kei and Yoshida, Kai and Yoshioka, Momoka and Yoshimura, Koei and Yamazaki, Koyo and Kiyotake, Shiko and Masuyama, Shota and Kobayashi, Ken and Nakata, Kazuhide},
  title     = {Relational Learning between Stores and Products Free from the Effects of Product Popularity and Store Scale in Retail Chains},
  booktitle = {Proceedings of the 2026 IEEE International Conference on Big Data (BigData)},
  year      = {2026}
}
```

## ライセンス

本リポジトリは [MIT License](LICENSE) の下で公開しています．
