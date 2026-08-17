#!/usr/bin/env python3
"""
Preprocesses raw text dataset, performs stratified splitting (80/10/10),
and serializes data into spaCy DocBin binary format for efficient training.
"""

import os
import argparse
import pandas as pd
import spacy
from tqdm import tqdm
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split


def create_docbin(df, nlp):
    """Converts a DataFrame of (text, label) into a spaCy DocBin."""
    db = DocBin()
    for text, label in tqdm(df[["text", "label"]].values, desc="Building DocBin"):
        doc = nlp.make_doc(str(text))
        if int(label) == 1:
            doc.cats = {"Hateful": 1, "Not-Hateful": 0}
        else:
            doc.cats = {"Hateful": 0, "Not-Hateful": 1}
        db.add(doc)
    return db


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for hate speech detection")
    parser.add_argument(
        "--input",
        type=str,
        default="sample_data/sample_hate_speech.csv",
        help="Path to cleaned input CSV dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docbins",
        help="Directory to save .spacy DocBin files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=108,
        help="Random seed for stratified splitting",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Total samples: {len(df)}")
    print(f"Class breakdown:\n{df['label'].value_counts(normalize=True)}")

    df["text"] = df["text"].astype("str")
    df["label"] = df["label"].astype("int")

    print("Splitting dataset (80% Train, 10% Dev, 10% Test)...")
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["label"]
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df["label"]
    )

    print(f"Train size: {len(train_df)}, Dev size: {len(dev_df)}, Test size: {len(test_df)}")

    nlp = spacy.blank("en")

    train_path = os.path.join(args.output_dir, "train.spacy")
    dev_path = os.path.join(args.output_dir, "dev.spacy")
    test_path = os.path.join(args.output_dir, "test.spacy")

    print(f"Saving {train_path}...")
    create_docbin(train_df, nlp).to_disk(train_path)

    print(f"Saving {dev_path}...")
    create_docbin(dev_df, nlp).to_disk(dev_path)

    print(f"Saving {test_path}...")
    create_docbin(test_df, nlp).to_disk(test_path)

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
