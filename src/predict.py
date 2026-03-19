"""
Inference script — predict bias label for any news headline.

Usage:
    python src/predict.py --headline "Senate blocks climate legislation"
    python src/predict.py --file my_headlines.txt
"""
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ID2LABEL = {0: "left", 1: "neutral", 2: "right"}
CONFIDENCE_THRESHOLD = 0.6


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def predict_one(headline: str, tokenizer, model, device: str) -> dict:
    inputs = tokenizer(headline, return_tensors="pt", truncation=True,
                        max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    low_conf = float(probs[pred_id]) < CONFIDENCE_THRESHOLD
    # If model is uncertain, fall back to neutral — ambiguous headlines are neutral
    final_label = "neutral" if low_conf else ID2LABEL[pred_id]
    return {
        "headline": headline,
        "prediction": final_label,
        "raw_prediction": ID2LABEL[pred_id],
        "confidence": float(probs[pred_id]),
        "probabilities": {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "low_confidence": low_conf,
    }


def predict_batch(headlines: list, tokenizer, model, device: str) -> list:
    return [predict_one(h, tokenizer, model, device) for h in headlines]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict media bias in news headlines")
    parser.add_argument("--model_path", default="models/roberta-bias/best_model")
    parser.add_argument("--headline", type=str, default=None, help="Single headline to classify")
    parser.add_argument("--file", type=str, default=None, help="Text file with one headline per line")
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model_path)
    print(f"Model loaded from: {args.model_path} | Device: {device}\n")

    if args.headline:
        r = predict_one(args.headline, tokenizer, model, device)
        print(f"Headline    : {r['headline']}")
        print(f"Prediction  : {r['prediction'].upper()}")
        if r["low_confidence"]:
            print(f"Raw model   : {r['raw_prediction'].upper()} (overridden to NEUTRAL — low confidence)")
        print(f"Confidence  : {r['confidence']:.1%}")
        print(f"Breakdown   : {r['probabilities']}")

    elif args.file:
        with open(args.file) as f:
            headlines = [line.strip() for line in f if line.strip()]
        results = predict_batch(headlines, tokenizer, model, device)
        print(f"{'LABEL':<10} {'CONF':<8} HEADLINE")
        print("-" * 70)
        for r in results:
            flag = " [!]" if r["low_confidence"] else ""
            print(f"{r['prediction'].upper():<10} {r['confidence']:.0%}     {r['headline']}{flag}")
    else:
        print("Provide --headline or --file. Example:")
        print('  python src/predict.py --headline "Congress passes new climate legislation"')
