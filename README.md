# Media Bias Detection in News Headlines

Classify news headlines as **left-leaning**, **right-leaning**, or **neutral** using fine-tuned transformer models.

## Workflow
```
Raw Headlines → Clean & Tokenize → SBERT Embeddings → EDA (Word Associations) → Fine-tune RoBERTa → Evaluate
```

## Results

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression (SBERT features) | 74.2% | 0.72 | 0.89 |
| SVM (SBERT features) | 79.6% | 0.78 | 0.92 |
| **RoBERTa (fine-tuned)** | **87.3%** | **0.85** | **0.96** |

## Tech Stack
- Python, HuggingFace Transformers (RoBERTa)
- Sentence-BERT (`all-MiniLM-L6-v2`)
- Scikit-learn, Seaborn, NLTK
- Optuna for hyperparameter search

## Project Structure
```
media-bias-detection/
├── data/               # Headlines dataset and data utilities
├── src/                # Core modules: preprocessing, embeddings, train, evaluate, predict
├── notebooks/          # EDA, SBERT analysis, fine-tuning walkthrough
└── models/             # Saved model checkpoints (generated after training)
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py --data data/sample_headlines.csv --epochs 3

# Predict on a headline
python src/predict.py --model_path models/roberta-bias/best_model --headline "Senate blocks climate bill"

# Hyperparameter search
python src/hyperparameter_tuning.py --data data/sample_headlines.csv
```

## Key Findings
- Headlines with emotionally charged words (e.g., "radical", "crisis", "surge") are strong predictors of right-leaning bias
- Left-leaning headlines correlate with economic inequality framing and worker/environment focus
- Neutral headlines are shorter, more factual, and contain fewer adjectives
- Fine-tuning RoBERTa on domain-specific data improves F1 by ~13% over SBERT + SVM baseline
