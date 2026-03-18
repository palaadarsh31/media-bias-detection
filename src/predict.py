"""
Inference script for media bias detection.
Usage:
    python src/predict.py --headline "Senate passes climate legislation"
    python src/predict.py --file headlines.txt
"""
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

ID2LABEL = {0: "left", 1: "neutral", 2: "right"}
CONFIDENCE_THRESHOLD = 0.6


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict_single(headline: str, tokenizer, model, device: str = "cpu") -> dict:
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return {
        "headline": headline,
        "prediction": ID2LABEL[pred_id],
        "confidence": float(probs[pred_id]),
        "probabilities": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
        "low_confidence": float(probs[pred_id]) < CONFIDENCE_THRESHOLD,
    }


def predict_batch(headlines: list, tokenizer, model, device: str = "cpu") -> list:
    return [predict_single(h, tokenizer, model, device) for h in headlines]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/roberta-bias/best_model")
    parser.add_argument("--headline", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)

    if args.headline:
        result = predict_single(args.headline, tokenizer, model)
        print(f"\nHeadline   : {result['headline']}")
        print(f"Prediction : {result['prediction'].upper()}")
        print(f"Confidence : {result['confidence']:.1%}")
        print(f"All probs  : {result['probabilities']}")
    elif args.file:
        with open(args.file) as f:
            headlines = [line.strip() for line in f if line.strip()]
        results = predict_batch(headlines, tokenizer, model)
        for r in results:
            flag = " [low confidence]" if r["low_confidence"] else ""
            print(f"[{r['prediction'].upper():<7}] ({r['confidence']:.0%}) {r['headline']}{flag}")
