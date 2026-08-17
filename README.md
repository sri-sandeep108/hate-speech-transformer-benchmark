# Comparative Analysis of Transformer Models and Static Word Embeddings for Hate Speech Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-v3.8-09a3d5.svg)](https://spacy.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **MSc in Data Analytics Dissertation**  
> **Author:** Sri Sandeep Sakthivel  
> **Full Dissertation:** [Dissertation.pdf](Dissertation.pdf)

---

## 📌 Executive Summary

The exponential growth of user-generated content on social media has made manual content moderation both unscalable and psychologically harmful to human moderators—who suffer high rates of secondary traumatic stress, burnout, and vicarious trauma. While automated Natural Language Processing (NLP) solutions are essential, platforms face a critical **Performance-Efficiency Dilemma**: heavy transformer architectures demand massive compute resources and introduce high latency, whereas lightweight static models struggle with context and subtle forms of hate speech.

This research presents a rigorous empirical benchmark comparing **five NLP architectures** across classification accuracy, minority class safety, and inference throughput on a unified ~446k sample corpus:
1. **Static Baseline:** GloVe vectors + Bi-LSTM (`en_core_web_lg`)
2. **Foundational Transformer:** BERT (`bert-base-uncased`, 110M params)
3. **Robustly Optimized Transformer:** RoBERTa (`roberta-base`, 125M params)
4. **Distilled Transformer:** DistilBERT (`distilbert-base-uncased`, 66M params)
5. **Discriminative Transformer:** ELECTRA (`google/electra-base-discriminator`, 110M params)

---

## 🏆 Key Findings

- **DistilBERT is the Optimal Production Model:** Contradicting the assumption that larger models always dominate, DistilBERT (66M parameters) achieved the highest **Macro F1-Score (0.8109)** and the highest **Hateful Recall (0.7165)**, while providing a **~17% speedup** (~41,238 words/sec) over standard transformer baselines.
- **Context is Critical:** The static GloVe embedding baseline missed **~40% of hateful content** (Recall: 0.6107), demonstrating that contextualized representations are mandatory for safety-critical moderation.
- **The Precision Ceiling (~0.67):** All transformer architectures plateaued around a precision ceiling of ~0.67–0.69. In fully autonomous deployment, **1 out of every 3 flagged posts would be a false positive**, risking severe wrongful censorship.
- **Recommended Deployment (Human-in-the-Loop):** Fully autonomous moderation is unsafe. Instead, a **confidence-thresholded triage pipeline** is proposed—automating unambiguous predictions while routing borderline samples to human reviewers, reducing human moderator trauma exposure by **50–60%**.

---

## 📊 Experimental Results

All models were evaluated on a held-out, stratified test set ($N \approx 44,600$ samples, 18% Hateful : 82% Non-Hateful).

| Architecture | Parameters | Macro F1 | Hateful Precision | Hateful Recall | Hateful F1 | Non-Hateful F1 | Throughput (Words/Sec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **DistilBERT** | **66M** | **0.8109** | **0.6687** | **0.7165** | **0.6918** | **0.9301** | **41,238** |
| 🥈 **ELECTRA** | 110M | 0.8060 | 0.6769 | 0.6867 | 0.6818 | 0.9302 | 34,802 |
| 🥉 **BERT** | 110M | 0.7993 | 0.6675 | 0.6737 | 0.6706 | 0.9280 | 35,184 |
| 🔹 **RoBERTa** | 125M | 0.7603 | 0.5808 | 0.6440 | 0.6108 | 0.9098 | 33,673 |
| 🔹 **Static Vector (GloVe)** | — | 0.7453 | 0.5621 | 0.6107 | 0.5854 | 0.9051 | **135,161** |

*Hardware Environment: Dedicated workstation featuring NVIDIA RTX 5090 (32GB VRAM), AMD EPYC (16 Cores / 32 Threads), 1TB DDR4 RAM, PyTorch 2.9, spaCy 3.8.*

---

## 📂 Repository Structure

```plaintext
├── Dissertation.pdf               # Complete Master's Dissertation (PDF)
├── configs/                       # spaCy pipeline training configurations
│   ├── base_config.cfg            # Base configuration for Static Vector (GloVe)
│   ├── base_config_gpu_bert.cfg   # Base config for BERT
│   ├── base_config_gpu_distilbert.cfg
│   ├── base_config_gpu_electra.cfg
│   ├── base_config_gpu_roberta.cfg
│   ├── gpu_bert.cfg               # Fully resolved training configs
│   ├── gpu_distilbert.cfg
│   ├── gpu_electra.cfg
│   ├── gpu_roberta.cfg
│   └── static_vector.cfg
├── docbins/                       # Binary spaCy DocBin datasets (train/dev/test)
│   └── .gitkeep
├── output/                        # Saved model checkpoints & best weights
│   └── .gitkeep
├── results/                       # Detailed JSON evaluation metrics per architecture
│   ├── bert
│   ├── distilbert
│   ├── electra
│   ├── roberta
│   └── static_vector
├── sample_data/                   # Stratified sample data for quick reproduction
│   └── sample_hate_speech.csv
├── scripts/                       # Modular CLI Python scripts
│   ├── preprocess.py              # Dataset cleaner & DocBin generator
│   ├── demo.py                    # Interactive terminal prediction tool
│   └── evaluate.py                # Results parser and table formatter
├── data_cleanup.ipynb             # Notebook: Token length filtering & text normalization
├── training.ipynb                 # Notebook: Stratified split & model training pipeline
├── benchmark.ipynb                # Notebook: spaCy evaluation benchmark execution
├── demo.ipynb                     # Notebook: Interactive HTML prediction interface
├── requirements.txt               # Python package dependencies
├── CITATION.cff                   # Citation metadata
├── citation.bib                   # BibTeX citation entry
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sri-sandeep108/hate-speech-transformer-benchmark.git
cd hate-speech-transformer-benchmark
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy base models
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf
```

---

## 🔄 Reproduction Pipeline

### Step 1: Preprocessing & DocBin Generation
To generate `.spacy` serialized DocBins from the dataset (or the included sample data):
```bash
python scripts/preprocess.py --input sample_data/sample_hate_speech.csv --output-dir docbins/
```

### Step 2: Training Models
Fill the configuration templates and initiate GPU training:
```bash
# Example: Training DistilBERT on GPU
python -m spacy init fill-config configs/base_config_gpu_distilbert.cfg configs/gpu_distilbert.cfg
python -m spacy train configs/gpu_distilbert.cfg \
    --paths.train docbins/train.spacy \
    --paths.dev docbins/dev.spacy \
    --output output/distilbert \
    --gpu-id 0
```

### Step 3: Benchmarking & Accuracy Evaluation
Evaluate trained model weights against the test partition:
```bash
python -m spacy benchmark accuracy output/distilbert/model-best/ docbins/test.spacy --output results/distilbert/ -g 0
```
Or display summary metrics across all architectures:
```bash
python scripts/evaluate.py
```

### Step 4: Interactive Model Inference
Run predictions across trained models via CLI:
```bash
# Direct input
python scripts/demo.py --text "I really love how supportive this online community is!"

# Interactive prompt mode
python scripts/demo.py
```
Or launch Jupyter Notebook for rich HTML rendering:
```bash
jupyter notebook demo.ipynb
```

---

## ⚖️ Ethical Considerations & Safety

1. **Algorithmic Bias & AAVE:** Automated filters risk misclassifying dialectal language (e.g., African American Vernacular English) used in reclaimed or benign contexts.
2. **Impact of False Positives:** With a ~33% false-positive rate, unchecked automated bans harm user speech and stifle legitimate discourse.
3. **Impact of False Negatives:** Missing ~29% of hateful content allows toxicity to spread, necessitating human oversight.
4. **Moderator Mental Health:** Human-in-the-loop filtering is designed not to eliminate human moderators, but to buffer them against the highest-volume trauma streams.

---

## 📖 Citation

If you find this research or code helpful in your work, please cite:

```bibtex
@mastersthesis{sakthivel2026comparative,
  author       = {Sri Sandeep Sakthivel},
  title        = {A Comparative Analysis of Transformer Models and Static Word Embedding Model for Hate Speech Detection},
  school       = {MSc in Data Analytics},
  year         = {2026},
  type         = {Master's Thesis}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
