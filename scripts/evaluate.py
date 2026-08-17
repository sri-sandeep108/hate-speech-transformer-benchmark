#!/usr/bin/env python3
"""
Summarizes and formats evaluation results from the results/ directory.
"""

import os
import json
import pandas as pd


RESULTS_DIR = "results"


def load_results():
    data = []
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory '{RESULTS_DIR}' not found.")
        return None

    for fname in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath, "r") as f:
                    metrics = json.load(f)

                model_name = fname.replace("_", " ").title()
                macro_f1 = metrics.get("cats_macro_f", metrics.get("cats_score", 0.0))
                h_p = metrics.get("cats_f_per_type", {}).get("Hateful", {}).get("p", 0.0)
                h_r = metrics.get("cats_f_per_type", {}).get("Hateful", {}).get("r", 0.0)
                h_f = metrics.get("cats_f_per_type", {}).get("Hateful", {}).get("f", 0.0)
                nh_p = metrics.get("cats_f_per_type", {}).get("Not-Hateful", {}).get("p", 0.0)
                nh_r = metrics.get("cats_f_per_type", {}).get("Not-Hateful", {}).get("r", 0.0)
                nh_f = metrics.get("cats_f_per_type", {}).get("Not-Hateful", {}).get("f", 0.0)
                speed = metrics.get("speed", 0.0)

                data.append(
                    {
                        "Architecture": model_name,
                        "Macro F1": round(macro_f1, 4),
                        "Hate Precision": round(h_p, 4),
                        "Hate Recall": round(h_r, 4),
                        "Hate F1": round(h_f, 4),
                        "Non-Hate Precision": round(nh_p, 4),
                        "Non-Hate Recall": round(nh_r, 4),
                        "Non-Hate F1": round(nh_f, 4),
                        "Speed (Words/Sec)": f"{int(round(speed)):,}",
                    }
                )
            except Exception as e:
                print(f"Error reading {fpath}: {e}")

    return pd.DataFrame(data)


def main():
    df = load_results()
    if df is not None and not df.empty:
        print("\n========================= EXPERIMENTAL BENCHMARK RESULTS =========================")
        print(df.to_string(index=False))
        print("==================================================================================\n")
    else:
        print("No evaluation results available.")


if __name__ == "__main__":
    main()
