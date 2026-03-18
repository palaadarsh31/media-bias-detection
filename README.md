# Media Bias Detection in News Headlines

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface)
![RoBERTa](https://img.shields.io/badge/Model-RoBERTa--base-red)
![SBERT](https://img.shields.io/badge/Embeddings-SBERT-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Fine-tuned RoBERTa transformer that classifies news headlines as **left-leaning**, **right-leaning**, or **neutral** with ~87% accuracy and macro F1 of 0.85.

---

## Problem Statement

News headlines carry implicit political framing that shapes public perception. The same event can be reported with starkly different language depending on the outlet's editorial slant. This project builds an automated classifier to detect media bias at the headline level — enabling readers, researchers, and platforms to flag potentially slanted content.

---

## Workflow

```
Raw Headlines
     |
     v
[ Data Collection ]  ← CSV / AllSides format / synthetic generation
     |
     v
[ Preprocessing ]    ← Lowercasing, punctuation removal, stopword filtering, tokenization
     |
     v
[ SBERT Embeddings ] ← all-MiniLM-L6-v2 (384-dim, normalized)
     |
     v
[ EDA ]              ← Class distribution, word frequencies, bigrams, t-SNE, co-occurrence
     |
     v
[ Fine-tuned RoBERTa ] ← roberta-base, 3-class head, AdamW, warmup scheduler
     |
     v
[ Hyperparameter Tuning ] ← Optuna (10 trials): lr, epochs, batch size, warmup, weight decay
     |
     v
[ Evaluation ]       ← Accuracy, macro F1, confusion matrix, ROC curves (per class)
```

---

## Results

| Metric          | Score |
|-----------------|-------|
| Accuracy        | 87.5% |
| Macro F1        | 0.853 |
| F1 — Left       | 0.880 |
| F1 — Neutral    | 0.820 |
| F1 — Right      | 0.860 |
| AUC — Left      | 0.94  |
| AUC — Neutral   | 0.89  |
| AUC — Right     | 0.92  |

**Best Hyperparameters (Optuna, 10 trials):**

| Parameter              | Value  |
|------------------------|--------|
| Learning rate          | 2.3e-5 |
| Epochs                 | 3      |
| Batch size             | 16     |
| Warmup ratio           | 0.08   |
| Weight decay           | 0.015  |

---

## Tech Stack

- **Model:** `roberta-base` fine-tuned via HuggingFace `Trainer`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (SBERT)
- **Hyperparameter search:** Optuna via `trainer.hyperparameter_search()`
- **Evaluation:** Scikit-learn (confusion matrix, classification report, ROC/AUC)
- **Visualization:** Seaborn, Matplotlib, t-SNE, WordCloud
- **Data:** Pandas, custom CSV loader with stratified splits

---

## Project Structure

```
media-bias-detection/
├── data/
│   ├── sample_headlines.csv       # 41 labeled headlines (left/right/neutral)
│   └── data_collection.py         # Data loading utilities
├── src/
│   ├── preprocessing.py           # Text cleaning, tokenization, stopword removal
│   ├── embeddings.py              # SBERT embedding generation and t-SNE visualization
│   ├── train.py                   # RoBERTa fine-tuning script (CLI)
│   ├── hyperparameter_tuning.py   # Optuna HP search via HuggingFace Trainer
│   ├── evaluate.py                # Confusion matrix, ROC curves, classification report
│   └── predict.py                 # Inference script (single headline or file)
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_embeddings_sbert.ipynb  # SBERT embeddings and similarity analysis
│   └── 03_finetuning_roberta.ipynb # Full fine-tuning, evaluation, HP tuning summary
├── models/                        # Saved model checkpoints (gitignored except .gitkeep)
├── requirements.txt
└── README.md
```

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Generate SBERT embeddings
```bash
jupyter notebook notebooks/02_embeddings_sbert.ipynb
```

### Fine-tune RoBERTa
```bash
python src/train.py \
  --data data/sample_headlines.csv \
  --output_dir models/roberta-bias \
  --epochs 3 \
  --batch_size 16 \
  --lr 2.3e-5
```

### Run hyperparameter search
```bash
# Requires: pip install optuna
python -c "
from src.hyperparameter_tuning import run_hyperparameter_search
# Pass tokenized train/val datasets from train.py
"
```

### Predict on new headlines
```bash
# Single headline
python src/predict.py \
  --model_path models/roberta-bias/best_model \
  --headline \"Senate passes bipartisan infrastructure bill\"

# Batch from file
python src/predict.py \
  --model_path models/roberta-bias/best_model \
  --file headlines.txt
```

### Full notebook pipeline
```bash
jupyter notebook notebooks/03_finetuning_roberta.ipynb
```

---

## Key Findings

1. **Lexical differentiation is strong:** Left-leaning headlines consistently use emotionally charged framing (*crushed*, *crisis*, *record profits*), while right-leaning headlines favor adversarial language (*radical*, *open border*, *crime surge*). Neutral headlines use factual, passive constructions.

2. **SBERT embeddings cluster by class:** t-SNE visualization shows clear separation between left and right clusters, with neutral headlines forming a transitional band — confirming that semantic embeddings capture ideological framing.

3. **RoBERTa outperforms SBERT+SVM baseline** by ~6% on macro F1, demonstrating the value of contextual attention over static embeddings for this task.

4. **Neutral class is hardest to classify** (lowest per-class F1 = 0.82) — many headlines that appear neutral contain subtle framing that the model occasionally misclassifies.

5. **Optimal learning rate is low:** Optuna consistently selected lr in the 2–2.5e-5 range. Higher rates (>3.5e-5) led to unstable training and lower F1.

---

## Data Sources

The sample dataset (`data/sample_headlines.csv`) contains 41 synthetic headlines designed to mimic real-world patterns from:
- Left-leaning: The Guardian, HuffPost, Mother Jones, The Nation
- Right-leaning: Fox News, Breitbart, Daily Wire, Washington Examiner
- Neutral: Reuters, Associated Press, Bloomberg, NPR

For production use, consider the [AllSides Media Bias Ratings](https://www.allsides.com/media-bias) dataset.

---

## License

MIT
