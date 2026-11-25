# npmi-hyperspherical-mf

[English](README.md) | 日本語

IEEE BigData 2026 採択論文 **「小売チェーンにおける商品人気・店舗規模の影響を排除した店舗・商品間の関係学習」** (*Relational Learning between Stores and Products Free from the Effects of Product Popularity and Store Scale in Retail Chains*) の公式実装リポジトリです．

- **著者**: Yoshimi Nakaya, Kei Yoshitake, Kai Yoshida, Momoka Yoshioka, Koei Yoshimura, Koyo Yamazaki, Shiko Kiyotake, Shota Masuyama, Ken Kobayashi, and Kazuhide Nakata (Institute of Science Tokyo)
- **発表会議**: IEEE International Conference on Big Data (IEEE BigData 2026)

---

## 概要

小売チェーンにおける店舗ごとの商品選定では，店舗と商品の潜在的な相性を定量化することが重要です．しかし，生の販売数に基づく既存の関係学習手法には以下の 3 つの構造的課題があります．

1. **人気バイアス**: 人気商品がすべての店舗と相性が良いと学習され，店舗固有の適性が見えにくくなる．
2. **店舗スケール問題**: 大型店舗の販売量の大きさに引きずられ，販売規模と相性の良否を区別できない．
3. **コールドスタート問題**: 購買履歴のない新規商品に対して埋め込みを生成できない．

本研究では，以下の 3 要素を統合した **超球面上の類似度学習フレームワーク** を提案します．

- **NPMI 変換 (前処理)**: 生の販売数の代わりに，周辺確率で正規化した Normalized Pointwise Mutual Information (NPMI, 値域 $[-1, 1]$) を教師データとして採用し，商品人気と店舗規模の影響を緩和する．
- **単位 $\ell_2$ ノルム制約**: 店舗・商品埋め込みベクトルに $\|\mathbf{u}_i\|_2 = \|\mathbf{v}_j\|_2 = 1$ を課し，内積をコサイン類似度 (値域 $[-1, 1]$) に一致させることで，教師データの値域と埋め込み空間の幾何的性質を整合させる．
- **補助情報制約**: 店舗特徴量 $\mathbf{X}_U$ と商品特徴量 $\mathbf{X}_V$ からエンコーダ $f_\theta, g_\phi$ を介して埋め込みを生成することで，購買履歴のない新規商品に対しても帰納的に埋め込みを生成可能にする．

### 定式化

購買行列 $\mathbf{R} \in \mathbb{Z}_{+}^{m \times n}$，観測マスク $\mathbf{M} \in \lbrace 0, 1 \rbrace^{m \times n}$，店舗補助情報 $\mathbf{X}_U \in \mathbb{R}^{m \times p}$，商品補助情報 $\mathbf{X}_V \in \mathbb{R}^{n \times q}$ に対して，観測ペアの NPMI 行列 $\mathbf{R}_{\text{NPMI}}$ を以下の制約付き最適化問題により近似します．

$$
\min_{\theta, \phi} \; \left\| \mathbf{M} \odot \left( \mathbf{R}_{\text{NPMI}} - f_\theta(\mathbf{X}_U) g_\phi(\mathbf{X}_V)^\top \right) \right\|_F^2
$$

ここで，各エンコーダ $f_\theta: \mathbb{R}^p \to \mathbb{R}^k$，$g_\phi: \mathbb{R}^q \to \mathbb{R}^k$ は出力層に $\ell_2$ 正規化層を備え，任意の $i \in \mathcal{I}, j \in \mathcal{J}$ に対して $\|f_\theta(\mathbf{x}^U_i)\|_2 = 1$，$\|g_\phi(\mathbf{x}^V_j)\|_2 = 1$ を満たします．

---

## リポジトリ構成

> **Note**: 本リポジトリは現在，カメラレディ論文の実験コードを整理・公開準備中です．最終的なディレクトリ構成は以下を予定しています．

```text
npmi-hyperspherical-mf/
├── configs/                # 実験設定ファイル (YAML: BookPOS / M5 Walmart, 提案手法・アブレーション・ベースライン)
├── data/                   # データ配置ディレクトリ (raw / processed は .gitignore 対象)
├── scripts/                # 前処理・学習・評価・集計・図表生成スクリプト
├── src/                    # モデル・学習器・評価指標の実装
│   ├── mf/                 #   提案手法 (NPMISphereModel), アブレーションモデル, ベースライン
│   ├── train_ablation.py   #   提案手法およびアブレーション条件の学習スクリプト
│   ├── train_baseline.py   #   ベースライン (DropoutNet, ContentConcat, StoreMean) の学習スクリプト
│   └── eval_unified.py     #   統一評価スクリプト (ランキング指標・人気バイアス指標・再構成指標)
├── tests/                  # 単体テスト
├── pyproject.toml          # 依存関係定義 (uv)
├── README.md               # 英語版 README
└── README_ja.md            # 本ファイル (日本語版 README)
```

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

論文中の全実験は NVIDIA H100 GPU 1 基を用いて実施しました．

---

## データセット

本論文では以下の 2 つのデータセットを用いて評価を行っています．

| データセット | $m$ (店舗数) | $n$ (商品数) | $|\Omega|$ (観測数) | $\rho$ (観測密度) | $\bar{r}$ (平均 NPMI) | $p$ (店舗特徴) | $q$ (商品特徴) | 公開可否 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **BookPOS** | 30 | 25,010 | 458,518 | 0.611 | 0.017 | 53 | 768 | 非公開 |
| **M5 Walmart** | 10 | 3,049 | 30,490 | 1.000 | $-0.007$ | 13 | 11 | 公開 |

### 1. M5 Walmart (再現可能)

公開データセットである [M5 Forecasting - Accuracy (Kaggle)](https://www.kaggle.com/c/m5-forecasting-accuracy) を使用します．元データをダウンロードして所定ディレクトリに配置したのち，前処理スクリプトを実行することで実験用行列 (`R`, `R_NPMI`, `mask`, `X_U`, `X_V`) と商品分割 (`train` / `val` / `test`) を構築できます．

- 店舗補助情報 $\mathbf{X}_U$ (13 次元): 所在州の one-hot (3 次元) + `train` 列の販売から算出した部門別・カテゴリ別の販売シェア (10 次元)．
- 商品補助情報 $\mathbf{X}_V$ (11 次元): カテゴリ・部門の one-hot ベクトル．

### 2. BookPOS (非公開データ)

日本の大手出版取次系列のチェーン書店 30 店舗における 2024 年 1 年間の日次書籍 POS データです．経営科学系研究部会連合協議会 (JASMAC) 2025 年度データ解析コンペティションにおいて株式会社トーハンより提供されたデータであり，データ利用規約のため **本リポジトリには含まれません**．

- 店舗補助情報 $\mathbf{X}_U$ (53 次元): 公式ウェブサイト・政府統計等から構築した店舗属性 (22 次元) + `train` 列の販売から算出した書籍大分類比率 (31 次元)．
- 書籍補助情報 $\mathbf{X}_V$ (768 次元): Google Books API から取得したあらすじを日本語 Sentence-BERT (`sonoisa/sentence-bert-base-ja-mean-tokens`) で埋め込み化したベクトル．

---

## 実験の再現手順

### 評価プロトコル (商品コールドスタート)

- 全商品 $n$ を 80% / 20% に分割し，20% をテスト用商品 $\mathcal{J}_{\text{test}}$ とします．さらに学習用商品の 80% / 20% を `train` / `val` に分割します．
- 学習に用いる NPMI は `train` 列の販売のみから周辺確率を推定し，`val` 列は `train` + `val` 列から，評価正解の `test` 列は全商品から算出します．
- 分割シードは固定し，学習シード `0`〜`9` の 10 試行で平均・標準偏差を算出します．

### 学習および評価の実行例

```bash
# 提案手法 (Full) の学習 (例: M5 Walmart, seed=0)
uv run python -m src.train_ablation \
  --config configs/m5/full.yaml \
  --seed 0 \
  --data-dir data/m5/processed \
  --split-parquet data/m5/data_split.parquet \
  --out outputs/m5/Full/seed_0

# ベースラインの学習 (例: DropoutNet, ContentConcat, StoreMean)
uv run python -m src.train_baseline \
  --config configs/m5/dropoutnet.yaml \
  --seed 0 \
  --data-dir data/m5/processed \
  --split-parquet data/m5/data_split.parquet \
  --out outputs/m5/DropoutNet/seed_0

# 統一評価の実行
uv run python -m src.eval_unified \
  --checkpoint outputs/m5/Full/seed_0 \
  --data-dir data/m5/processed \
  --split-parquet data/m5/data_split.parquet \
  --condition Full \
  --seed 0 \
  --out results/m5/Full/seed_0
```

### 主なハイパーパラメータ

| パラメータ | 値 |
| :--- | :--- |
| 店舗側エンコーダ $f_\theta$ | バイアス項なしの線形写像 + $\ell_2$ 正規化 |
| 商品側エンコーダ $g_\phi$ | 2 層 MLP (隠れ次元 64) + $\ell_2$ 正規化 |
| 埋め込み次元 $k$ | 32 |
| 最適化手法 | Adam (学習率 $\eta = 10^{-3}$，フルバッチ) |
| 重み減衰 (weight decay) | $10^{-4}$ (`wo_norm`・`wo_both` は $0$) |
| 最大エポック数 / 早期終了 | $10^5$ / 検証 MSE で $\tau_{\text{stop}} = 500$ エポック改善なし |

---

## 主な実験結果

すべての数値は 10 seeds (`0`〜`9`) の平均 $\pm$ 標準偏差 ($K=10$) です．太字は Random を除く各列の最良値を示します．

### BookPOS 主結果

| 手法 | Prec@10 | NDCG@10 | MAP@10 | Personalization@10 | $\rho_{\text{store}}$ | $R^2_{\text{test}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **提案手法** | **0.596 ± 0.040** | **0.602 ± 0.040** | **(2.33 ± 0.22) $\times 10^{-3}$** | **0.968 ± 0.006** | 0.200 ± 0.004 | 0.000 ± 0.007 |
| DropoutNet | 0.522 ± 0.014 | 0.510 ± 0.016 | (1.86 ± 0.10) $\times 10^{-3}$ | 0.873 ± 0.021 | 0.221 ± 0.010 | 0.109 ± 0.008 |
| ContentConcat | 0.463 ± 0.038 | 0.445 ± 0.038 | (1.54 ± 0.20) $\times 10^{-3}$ | 0.843 ± 0.031 | **0.246 ± 0.005** | **0.123 ± 0.002** |
| StoreMean | - | - | - | - | - | 0.063 ± 0.000 |
| CategoryPop | 0.387 ± 0.026 | 0.388 ± 0.029 | (1.22 ± 0.13) $\times 10^{-3}$ | 0.000 ± 0.000 | $-0.024 \pm 0.000$ | - |
| Random | 0.377 ± 0.028 | 0.377 ± 0.031 | (1.12 ± 0.14) $\times 10^{-3}$ | 0.999 ± 0.000 | 0.000 ± 0.003 | - |

### M5 Walmart 主結果

| 手法 | Prec@10 | NDCG@10 | Personalization@10 | $\rho_{\text{store}}$ | $R^2_{\text{test}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **提案手法** | **0.746 ± 0.015** | **0.743 ± 0.024** | 0.956 ± 0.008 | **0.309 ± 0.005** | 0.160 ± 0.003 |
| DropoutNet | 0.651 ± 0.066 | 0.635 ± 0.064 | 0.928 ± 0.023 | 0.282 ± 0.012 | 0.129 ± 0.005 |
| ContentConcat | 0.683 ± 0.051 | 0.689 ± 0.068 | **0.966 ± 0.017** | 0.298 ± 0.005 | **0.170 ± 0.003** |
| StoreMean | - | - | - | - | 0.042 ± 0.000 |
| CategoryPop | 0.442 ± 0.033 | 0.443 ± 0.036 | 0.000 ± 0.000 | $-0.002 \pm 0.000$ | - |
| Random | 0.439 ± 0.048 | 0.439 ± 0.052 | 0.991 ± 0.003 | 0.001 ± 0.013 | - |

### アブレーション結果 (BookPOS)

| 条件 | Prec@10 | NDCG@10 | head-NDCG@10 | tail-NDCG@10 | MAP@10 | Personalization@10 | APLT@10 | MAE$_{\text{test}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **提案手法** | **0.596 ± 0.040** | **0.602 ± 0.040** | **0.705 ± 0.024** | **0.575 ± 0.035** | **(2.33 ± 0.22) $\times 10^{-3}$** | 0.968 ± 0.006 | **0.841 ± 0.022** | **0.042 ± 0.000** |
| NPMI なし | 0.460 ± 0.026 | 0.459 ± 0.031 | 0.536 ± 0.020 | 0.409 ± 0.021 | (1.54 ± 0.14) $\times 10^{-3}$ | 0.965 ± 0.011 | 0.600 ± 0.045 | 0.095 ± 0.004 |
| ノルム制約なし | 0.567 ± 0.037 | 0.568 ± 0.038 | 0.651 ± 0.023 | 0.550 ± 0.030 | (2.13 ± 0.21) $\times 10^{-3}$ | **0.985 ± 0.004** | 0.824 ± 0.025 | 0.044 ± 0.000 |
| 両方なし | 0.432 ± 0.019 | 0.438 ± 0.023 | 0.545 ± 0.026 | 0.386 ± 0.028 | (1.41 ± 0.11) $\times 10^{-3}$ | 0.965 ± 0.012 | 0.681 ± 0.029 | 0.175 ± 0.006 |

### アブレーション結果 (M5 Walmart)

| 条件 | Recall@10 | MAP@10 | NDCG@10 |
| :--- | :---: | :---: | :---: |
| **提案手法** | **0.0295 ± 0.0007** | **0.0245 ± 0.0010** | **0.7435 ± 0.0243** |
| NPMI なし | 0.0275 ± 0.0010 | 0.0228 ± 0.0013 | 0.7258 ± 0.0243 |
| ノルム制約なし | 0.0274 ± 0.0017 | 0.0223 ± 0.0029 | 0.7026 ± 0.0545 |
| 両方なし | 0.0271 ± 0.0011 | 0.0235 ± 0.0011 | 0.7432 ± 0.0213 |

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

---

## 謝辞

本研究は，経営科学系研究部会連合協議会 (Joint Association Study Group of Management Science) 主催のデータ解析コンペティションの一環として実施されました．運営関係者の皆様ならびにデータをご提供いただいた株式会社トーハンに感謝申し上げます．

## ライセンス

本リポジトリのライセンスは確定次第 `LICENSE` ファイルに記載します．
