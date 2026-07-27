# FMVM-JITDP: Fusing Multi-View Code Change Representations for JIT Defect Prediction

[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports-b31b1b.svg)](#citation)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation and dataset for **FMVM-JITDP**, a multi-view fusion model for
Just-in-Time (JIT) software defect prediction. FMVM-JITDP represents each code change from
three complementary perspectives — a **TF-IDF lexical-change delta**, **node-type-level
AST structural-change statistics**, and **sixteen code churn metrics** — and fuses them with
a hybrid **BiLSTM + Transformer** architecture and a **commit-conditional attention** layer.

---

## Overview

Most JIT defect prediction methods observe a commit through a single lens (process, lexical,
or structural), leaving complementary evidence about *what* changed, *how* it changed, and
*how often* the code has changed unexploited. FMVM-JITDP closes three gaps identified in the
paper: (i) single-view blindness, (ii) two-view fusion that is static and post-commit, and
(iii) heterogeneous features forced through a single encoder.

<p align="center">
  <img src="assets/framework.png" alt="Overview of the proposed FMVM-JITDP framework" width="95%">
</p>

<p align="center">
  <em><b>Figure 1.</b> Overview of the proposed FMVM-JITDP framework. Three views (AST-based
  structural change, TF-IDF lexical change, and process/churn metrics) are routed through two
  branch-specific encoders — a Transformer branch over the concatenated 331-d code-change
  vector and a two-layer BiLSTM branch over the 16 churn metrics — and combined by a gated
  weighted-fusion layer before the prediction head.</em>
</p>

> **Adding the figure:** place your high-resolution Figure 1 at `assets/framework.png`
> (PNG or SVG). GitHub renders it inline once the file is committed at that path. If you
> prefer a different location or filename, update the `src` in the two `<img>`/`![]()`
> references above.

---

## Key Contributions

1. **Three-view representation.** Each change is described by a TF-IDF lexical delta
   (`d_tfidf = 300`), node-type-level AST edit-script statistics (`d_ast = 31`), and 16 code
   churn metrics — all formulated as *differences* between pre- and post-commit revisions so
   the classifier receives an explicit signal of what changed.
2. **Hybrid two-branch encoder.** A two-layer BiLSTM models interactions among the churn
   metrics; a two-block Transformer processes the concatenated 331-d code-change vector. The
   branches share no parameters, respecting the differing geometries of the two modalities.
3. **Commit-conditional gated fusion.** A softmax attention layer produces per-commit weights
   (α_m, α_c) over the metric and code-change branches, downweighting high-churn/low-lexical
   changes on the code branch and vice versa — behaviour static fusion cannot express.
4. **Comprehensive evaluation.** On nine open-source projects, FMVM-JITDP outperforms seven
   state-of-the-art baselines on all six indicators, with significance confirmed by
   Wilcoxon (Holm–Bonferroni corrected), Cohen's d_z, and Friedman/Nemenyi tests.

---

## Repository Structure

```
FMVMJITDP/
├── assets/
│   └── framework.png          # Figure 1 (place your high-res figure here)
├── data/                      # Selected-release datasets + matched source files
│   ├── activemq/ camel/ ...   # one folder per project (see Dataset section)
│   └── README.md
├── src/
│   ├── feature_tfidf.py       # TF-IDF delta extraction (Eq. 5)
│   ├── feature_ast.py         # AST edit-script + 31-d histogram (Algorithm 1)
│   ├── feature_churn.py       # 16 code churn metrics (Table 1)
│   └── preprocess.py          # deterministic preprocessing / dedup (Table 2)
├── LSTMTransform.py           # FMVM-JITDP model: BiLSTM + Transformer + gated fusion
├── train.py                   # 10-run stratified 80/20 protocol, threshold selection
├── evaluate.py                # metrics + confusion counts (Table 8)
├── stats/
│   ├── pairwise_tests.py      # Wilcoxon + Holm–Bonferroni + Cohen's d_z (Table 9)
│   └── friedman_nemenyi.py    # Friedman omnibus + Nemenyi CD (Table 10)
├── results/                   # generated metrics tables and figures
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

> Adjust the file names above to match your working tree (e.g. `LSTMTransform.py` is the
> model definition). The structure is a suggested layout, not a hard requirement.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/ahmedabd39/FMVMJITDP.git
cd FMVMJITDP

# 2. (Recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Experiments in the paper were run on Google Colab with an NVIDIA T4 GPU, dual CPU, 12.7 GB
RAM, and 78.2 GB storage. A GPU is recommended but not required.

---

## Dataset

FMVM-JITDP is evaluated on the benchmark defect dataset of Yatish et al. (ICSE 2019), further
examined by Wattanakriengkrai et al. (TSE 2020), using **one selected release from each of
nine open-source projects**. For each labeled code-change instance the lexical and AST views
are built from the pre- and post-change states surrounding the corresponding commit.

| Project (release)     | Domain                          | Retained | Defective rate (%) |
|-----------------------|---------------------------------|---------:|-------------------:|
| ActiveMQ (5.2.0)      | Messaging / integration server  |    2,029 |              10.45 |
| Camel (2.10.0)        | Enterprise integration          |    7,896 |               2.82 |
| Derby (10.3.1.4)      | Relational database             |    2,067 |              27.00 |
| Groovy (1.6.Beta_1)   | JVM programming language        |      809 |               8.03 |
| HBase (0.95.2)        | Distributed data store          |    1,720 |              24.53 |
| Hive (0.9.0)          | Hadoop data warehouse           |    1,378 |              18.58 |
| JRuby (1.7.0)         | Ruby for the JVM                |    1,553 |               3.80 |
| Lucene (2.3.0)        | Text-search library             |      797 |              23.71 |
| Wicket (1.5.3)        | Web-application framework       |    2,567 |               3.93 |

The dataset and its matched source-code files are released with this repository. See
`data/README.md` for the per-project folder layout and column descriptions.

---

## Usage

```bash
# 1. Build the three views for a project (TF-IDF delta, AST histogram, churn metrics)
python src/preprocess.py --project derby

# 2. Train FMVM-JITDP with the 10-run stratified 80/20 protocol
python train.py --project derby --runs 10 --seed 42

# 3. Evaluate: F1, AUC, MCC, PofB20, Popt, BA + confusion counts
python evaluate.py --project derby

# 4. Statistical comparison against baselines (all nine projects)
python stats/pairwise_tests.py       # Wilcoxon / Holm–Bonferroni / Cohen's d_z
python stats/friedman_nemenyi.py     # Friedman omnibus + Nemenyi critical difference
```

**Protocol notes (matching the paper).** All preprocessing quantities — TF-IDF vocabulary
and IDF statistics, z-score parameters, class weights — are fitted on the 80% training subset
only. The operating threshold τ is selected on the 20% held-out subset by maximizing F1.
Weighted binary cross-entropy is used by default; focal loss (γ = 2.0) is applied when the
training-subset defective rate is below 5%. The run attaining the highest F1 is retained per
model–project combination, keeping threshold-dependent indicators consistent with the reported
confusion matrix.

---

## Model Configuration

| Component            | Setting |
|----------------------|---------|
| TF-IDF               | `max_features=300`, `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95` |
| AST feature vector   | 31-d (29 node-type counts + total nodes + node variety); similarity θ = 0.7 |
| Code-change vector   | Concatenation of TF-IDF delta (300) + AST vector (31) → `d_code = 331`, z-scored |
| BiLSTM branch        | BiLSTM(64, return_sequences=True, dropout=0.2) → BiLSTM(32, dropout=0.1); output 64-d |
| Transformer branch   | Dense(6400)→reshape(100×64), sinusoidal PE, 2 blocks (heads=4, ff=128, dropout=0.1); GAP → 64-d |
| Fusion               | Dense(2, softmax) gate → weighted concatenation → 128-d |
| Head                 | Dense(128)→Dense(64)→Dense(32)→Dense(1, sigmoid), BatchNorm + dropout |
| Optimizer            | Adam, lr = 0.001 (BCE) / 0.0005 (focal loss) |
| Training             | batch = 32, ≤100 epochs, early stopping (patience 15) on the 20% held-out subset |

Total trainable parameters ≈ 2.30M (the dense projection is ≈ 92% of the code branch).

---

## Results (project-averaged over nine projects)

| Method        |    F1 |   AUC |   MCC | PofB20 |  Popt |    BA |
|---------------|------:|------:|------:|-------:|------:|------:|
| **FMVM-JITDP**| **0.531** | **0.862** | **0.476** | **0.788** | **0.894** | **0.784** |
| JIT-CF        | 0.501 | 0.819 | 0.442 |  0.669 | 0.833 | 0.775 |
| MTL-DNN       | 0.498 | 0.807 | 0.440 |  0.699 | 0.837 | 0.744 |
| CodeT5        | 0.485 | 0.817 | 0.417 |  0.693 | 0.855 | 0.742 |
| MOJ-SDP       | 0.468 | 0.817 | 0.408 |  0.698 | 0.873 | 0.742 |
| CodeBERT      | 0.462 | 0.794 | 0.395 |  0.665 | 0.853 | 0.738 |
| GH-ACE        | 0.458 | 0.778 | 0.397 |  0.682 | 0.846 | 0.721 |
| EATT          | 0.450 | 0.793 | 0.386 |  0.674 | 0.863 | 0.745 |

FMVM-JITDP attains the best score on all six indicators and the best average Friedman rank on
each. Across the 54 project–indicator cells it is strictly best in 44 and tied in one; after
Holm–Bonferroni correction the improvement is significant (α = 0.05) in 38 of 42 pairwise
comparisons, the four exceptions all on balanced accuracy. Full per-project results (Precision,
Recall, and TP/FP/FN/TN confusion counts) are in the paper's Table 8 and under `results/`.

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{abdu2026fmvmjitdp,
  title   = {FMVM-JITDP: Fusing Multi-View Code Change Representations for JIT Defect Prediction},
  author  = {Abdu, Ahmed and Yumbla, Francisco and Al-Mahbashi, Mohammed and Algabri, Redhwan},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Under revision. Update volume/pages/DOI upon acceptance.}
}
```

---

## License

This project is released under the MIT License — see [LICENSE](LICENSE).

## Contact

- **Ahmed Abdu** — School of Computer Science, Shandong Xiehe University, Jinan, China
- **Redhwan Algabri** (corresponding) — Sejong University, Seoul, Republic of Korea · `redhwan@sejong.ac.kr`
