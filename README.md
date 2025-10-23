# Fake News Detection using RoBERTa + LoRA Fine-Tuning

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Active-success.svg)

A robust **Fake News Detection** system built using **RoBERTa** with **LoRA (Low-Rank Adaptation)** fine-tuning.
This project fine-tunes large transformer models efficiently to detect misinformation across multiple datasets,
focusing on **cross-domain generalization** between **FakeNews-Kaggle** and **LIAR** datasets.

---

## Table of Contents

* [Overview](#overview)
* [Project Goals](#project-goals)
* [Pipeline Overview](#pipeline-overview)
* [Repository Structure](#repository-structure)
* [Setup Instructions](#setup-instructions)
* [Usage](#usage)
* [Results](#results)
* [Key Learnings](#key-learnings)
* [Future Improvements](#future-improvements)
* [License](#license)

---

## Overview

This project explores fine-tuning of transformer-based models (RoBERTa) for fake news classification
using lightweight **LoRA adapters** to reduce computational cost while maintaining accuracy.

It combines multiple datasets—**FakeNews-Kaggle** and **LIAR**—to evaluate both **in-domain** and **cross-domain** generalization.
The implementation emphasizes reproducibility, modularity, and clean experimentation practices.

---

## Project Goals

* Build a **reproducible end-to-end pipeline** for fake news detection
* Fine-tune RoBERTa efficiently using **PEFT (Parameter-Efficient Fine-Tuning)**
* Achieve **high F1-score** on FakeNews dataset with limited compute
* Test **cross-domain robustness** using the LIAR dataset
* Provide clear evaluation, visualization, and extendability for new datasets

---

## ️ Pipeline Overview

| Stage                | Description                                                                       |
| -------------------- | --------------------------------------------------------------------------------- |
| **1. Data Fetching** | Automatically downloads, cleans, and structures datasets (FakeNews-Kaggle, LIAR). |
| **2. Preprocessing** | Normalizes text, handles class imbalance, tokenizes using RoBERTa tokenizer.      |
| **3. Model Setup**   | Initializes RoBERTa base model with LoRA adapters (via `peft` library).           |
| **4. Training**      | Fine-tunes using mixed precision (AMP) with progress tracking and checkpointing.  |
| **5. Evaluation**    | Computes accuracy, precision, recall, and F1-score across domains.                |
| **6. Visualization** | Generates performance summaries and confusion matrices.                           |

---

##  Repository Structure

```
FakeNews-Detection/
│
├── src/
│   ├── data/
│   │   ├── fetch_datasets.py          # Downloads and processes the LIAR and FakeNews datasets
│   │   ├── datasets.py                # Custom PyTorch Dataset class for tokenized inputs
│   │   └── preprocess.py              # Text normalization, deduplication, and label encoding
│   │
│   ├── models/
│   │   ├── text_model.py              # RoBERTa + LoRA model architecture
│   │   ├── train_utils.py             # Training utilities and checkpoint saving
│   │   └── eval_utils.py              # Evaluation metrics and reporting helpers
│   │
│   └── train.py                       # CLI entry point for fine-tuning and training
│
├── notebooks/
│   ├── 1_dataset_exploration.ipynb    # Data analysis and summary visualization
│   ├── 2_training_lora.ipynb          # LoRA fine-tuning pipeline with progress tracking
│   ├── 3_evaluation.ipynb             # Model evaluation and cross-domain testing
│   └── 4_visualization.ipynb          # Performance visualization and explainability (post-SHAP)
│
├── data/
│   ├── raw/                           # Raw datasets (auto-downloaded)
│   └── processed/                     # Cleaned, tokenized, and split data
│
├── models/
│   ├── roberta_lora_multidomain_best/   # Best-performing fine-tuned model checkpoint
│   ├── roberta_lora_multidomain_steps/  # Step-wise checkpoints during training
│   ├── roberta_lora_multidomain_merged/ # Adapter-merged model for export or inference
│   └── roberta_lora_multidomain_history.csv  # Training metrics log
│
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md                          # Project documentation
```

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/FakeNews-Detection.git
cd FakeNews-Detection
```

### 2. Create and Activate the Environment

```bash
conda create -n fakenews python=3.10 -y
conda activate fakenews
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Dataset Download and Processing

```bash
python -m src.data.fetch_datasets
```

This command automatically fetches the LIAR and FakeNews-Kaggle datasets, cleans them,
and stores processed CSV files in the `data/processed/` folder.

---

##  Usage

### 1. Dataset Exploration

Explore class balance, average text lengths, and data quality:

```bash
jupyter notebook notebooks/1_dataset_exploration.ipynb
```

### 2. Fine-Tune the Model

Train the LoRA-adapted RoBERTa model across domains:

```bash
python -m src.train --model roberta-base --lora --datasets FakeNews LIAR
```

You can adjust hyperparameters directly in the notebook `2_training_lora.ipynb`.

### 3. Evaluate the Model

```bash
jupyter notebook notebooks/3_evaluation.ipynb
```

This notebook loads the best-performing checkpoint, evaluates across both datasets,
and generates a detailed classification report and metric table.


---

##  Results Summary

| Dataset      | Accuracy | Precision | Recall | F1     |
| ------------ | -------- | --------- | ------ | ------ |
| **FakeNews** | 0.7729   | 0.8458    | 0.7729 | 0.7633 |
| **LIAR**     | 0.4431   | 0.7535    | 0.4431 | 0.2731 |

### Interpretation

* Strong, consistent performance on **FakeNews-Kaggle** with balanced precision–recall.
* Limited cross-domain generalization on **LIAR** due to short-form factual claims.
* Model effectively learns **contextual and stylistic** signals of misinformation.

---

##  Key Learnings

* **LoRA fine-tuning** reduces GPU memory requirements and speeds up training significantly.
* **Domain alignment** matters: models trained on article-level data struggle on short claims.
* **Mixed precision (AMP)** offers substantial performance gains on CUDA-enabled systems.
* Clean preprocessing and text normalization critically affect cross-dataset results.

---

##  Future Improvements

* Integrate **domain-adaptive fine-tuning** for short factual claims (LIAR).
* Add **multi-dataset merging** and balanced sampling utilities.
* Introduce a lightweight **inference API** using `FastAPI` for real-world testing.

---

##  License

This project is licensed under the [MIT License](LICENSE).
You’re free to use, modify, and distribute this repository for academic or research purposes.

---

##  Acknowledgments

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
* [PyTorch](https://pytorch.org/)
* [FakeNews-Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* [LIAR Dataset](https://aclanthology.org/P17-2067/)

---

