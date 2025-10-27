# Fake News Detection using RoBERTa + LoRA Fine-Tuning

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Active-success.svg)

A high-performance **Fake News Detection** pipeline built on **RoBERTa** with **LoRA (Low-Rank Adaptation)** fine-tuning. The project demonstrates how parameter-efficient fine-tuning can deliver strong results even on limited compute, with a focus on **cross-domain generalization** across **FakeNews-Kaggle** and **LIAR** datasets.

---

## Table of Contents

- [Overview](#overview)
- [Project Goals](#project-goals)
- [Pipeline Overview](#pipeline-overview)
- [Repository Structure](#repository-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Key Insights](#key-insights)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

This project applies **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA** adapters on **RoBERTa-base**, enabling efficient adaptation of large language models for misinformation detection. It evaluates both **in-domain performance** (FakeNews-Kaggle) and **out-of-domain generalization** (LIAR dataset) to measure model robustness.

---

## Project Goals

- Develop an **end-to-end reproducible pipeline** for fake news detection.
- Fine-tune **RoBERTa** efficiently using **LoRA adapters**.
- Benchmark **cross-domain performance** using both FakeNews and LIAR datasets.
- Maintain lightweight training with **minimal GPU memory footprint**.
- Provide structured evaluation metrics and results visualization.

---

## Pipeline Overview

| Stage | Description |
|--------|--------------|
| **1. Data Fetching** | Downloads and preprocesses FakeNews-Kaggle and LIAR datasets. |
| **2. Preprocessing** | Cleans text, encodes labels, and prepares train/val/test splits. |
| **3. Model Setup** | Initializes RoBERTa-base with LoRA adapters using the PEFT library. |
| **4. Training** | Fine-tunes model using mixed precision and checkpoint saving. |
| **5. Evaluation** | Computes metrics (Accuracy, Precision, Recall, F1) across domains. |

---

## Repository Structure

```bash
FakeNews-Detection/
├── src/
│   ├── data/
│   │   ├── fetch_datasets.py       # Fetch and prepare datasets (FakeNews, LIAR)
│   │   ├── datasets.py             # PyTorch Dataset class
│   │   └── preprocess.py           # Cleaning, tokenization, and encoding
│   │
│   ├── models/
│   │   ├── text_model.py           # RoBERTa + LoRA architecture
│   │   └── train.py                # Training loop and checkpoint saving
│   │
│   ├── evaluate.py                 # Evaluation script for performance metrics
│   └── utils/
│       ├── metrics.py              # Metric computation utilities
│       └── seed.py                 # Reproducibility and seed setup
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Dataset analysis and visualization
│   ├── 02_model_training.ipynb     # LoRA fine-tuning workflow
│   └── 03_model_evaluation.ipynb   # Performance evaluation and confusion matrix
│
├── data/
│   ├── raw/                        # Raw data (downloaded automatically)
│   └── processed/                  # Cleaned and tokenized data splits
│
├── models/
│   ├── roberta_lora_multidomain_best/   # Best-performing model checkpoint
│   ├── roberta_lora_multidomain_steps/  # Step-based checkpoints
│   ├── roberta_lora_multidomain_merged/ # Final merged adapter model
│   ├── roberta_lora_multidomain_history.csv # Training history log
│   └── roberta_lora_multidomain_evaluation_summary.csv # Evaluation summary
│
├── requirements.txt                # Python dependencies
├── Makefile                        # Reproducible training commands
└── README.md                       # Project documentation
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ashbix23/Misinformation-Classifier.git
cd Misinformation-Classifier
```

### 2. Create a Virtual Environment

```bash
conda create -n fakenews python=3.10 -y
conda activate fakenews
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and Preprocess Data

```bash
python -m src.data.fetch_datasets
```

This script downloads, cleans, and processes both FakeNews-Kaggle and LIAR datasets, saving ready-to-use files in `data/processed/`.

---

## Usage

### 1. Explore the Data

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Train the Model

```bash
python -m src.train --model roberta-base --lora --datasets FakeNews LIAR
```

### 3. Evaluate the Model

```bash
jupyter notebook notebooks/03_model_evaluation.ipynb
```

This notebook generates accuracy, precision, recall, F1, and confusion matrices for each dataset.

---

## Results

| Dataset | Accuracy | Precision | Recall | F1 |
|----------|-----------|------------|---------|---------|
| **FakeNews** | 0.773 | 0.846 | 0.773 | 0.763 |
| **LIAR** | 0.443 | 0.754 | 0.443 | 0.273 |

### Interpretation

- Excellent **in-domain** performance on FakeNews-Kaggle.
- Significant drop in **cross-domain generalization** for LIAR (short factual claims).
- Model learns strong **contextual and linguistic cues** for misinformation.

---

## Key Insights

- **LoRA fine-tuning** reduces memory and compute cost while preserving model quality.
- **Cross-domain adaptation** remains challenging due to dataset structural differences.
- **Clean preprocessing and normalization** dramatically improve results.
- **Mixed Precision Training (AMP)** enhances training efficiency on GPUs.

---

## Future Work

- Domain-adaptive fine-tuning for short-form factual claims.
- Add more datasets for multi-domain robustness.
- Build an **inference API** using FastAPI or Gradio for quick model testing.

---

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it for academic or research purposes.

---

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [PyTorch](https://pytorch.org/)
- [FakeNews-Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [LIAR Dataset](https://aclanthology.org/P17-2067/)


