#!/usr/bin/env python3
"""
Interactive CLI demo for Hate Speech Detection across trained models.
"""

import os
import sys
import argparse
import spacy


MODEL_PATHS = {
    "Static Vector (GloVe)": "output/static_vector/model-best",
    "BERT": "output/bert/model-best",
    "RoBERTa": "output/roberta/model-best",
    "DistilBERT": "output/distilbert/model-best",
    "ELECTRA": "output/electra/model-best",
}


def load_models():
    models = {}
    print("Loading available models from output/ ...")
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                models[name] = spacy.load(path)
                print(f"  [+] Loaded {name}")
            except Exception as e:
                print(f"  [-] Failed to load {name}: {e}")
        else:
            print(f"  [!] Skipping {name} (path {path} not found)")
    return models


def predict(text, models):
    print("\n" + "=" * 60)
    print(f"Input Text: \"{text}\"")
    print("=" * 60)
    print(f"{'Model':<25} | {'Prediction':<12} | {'Hate Score':<12} | {'Non-Hate Score':<14}")
    print("-" * 60)

    for name, nlp in models.items():
        doc = nlp(text)
        h_score = doc.cats.get("Hateful", 0.0)
        nh_score = doc.cats.get("Not-Hateful", 0.0)
        pred = "Hateful" if h_score > nh_score else "Not-Hateful"

        print(f"{name:<25} | {pred:<12} | {h_score:<12.4f} | {nh_score:<14.4f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive Hate Speech Classification Demo")
    parser.add_argument("--text", type=str, default=None, help="Text to classify directly")
    args = parser.parse_args()

    models = load_models()
    if not models:
        print("No models available. Please train models or download pre-trained weights to output/ directory.")
        sys.exit(1)

    if args.text:
        predict(args.text, models)
        return

    print("\nEnter a sentence to classify (or type 'exit' / 'quit' to stop):")
    while True:
        try:
            user_input = input("\n>> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting demo.")
                break
            predict(user_input, models)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo.")
            break


if __name__ == "__main__":
    main()
