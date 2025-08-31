# **TransLlamaX: Hinglish → English Translation System**

TransLlamaX is an **end-to-end machine translation pipeline** for Hinglish → English, leveraging **LLaMA-3-8B** with advanced fine-tuning techniques. It is designed to handle **informal, social-media style text** with mixed Hindi-English romanized input. The system integrates **LoRA/QLoRA, supervised fine-tuning (SFT), preference optimization (RLHF), evaluation, and explainability**.

---

## **Features**

* **State-of-the-art fine-tuning** of LLaMA-3-8B:

  * Low-rank adaptation (LoRA/QLoRA)
  * Supervised fine-tuning (SFT)
  * Preference optimization / RLHF for fluency & adequacy
* **Robust evaluation** across public and custom datasets:

  * Metrics: BLEU, METEOR, ROUGE-L, GLEU, BERTScore-F1, BLEURT-20
  * Statistical tests: Wilcoxon signed-rank, Friedman omnibus, Conover–Holm post-hoc
* **Explainability**:

  * LIME token-level analysis
  * SHAP feature importance analysis
* **Handles mixed-script and informal text** typical of social media / creator content.
* **Fully modular Python package** — no notebooks.

---

## **Repository Structure**

```
TransLlamaX/
│
├── data/
│   ├── raw/                     # Original datasets (FindNITAI, custom)
│   ├── processed/               # Tokenized / cleaned datasets
│   └── utils.py                  # Preprocessing & tokenization
│
├── models/
│   ├── base_model.py             # Load LLaMA-3-8B or baselines
│   ├── train_model.py            # LoRA/QLoRA SFT + RLHF fine-tuning
│   ├── inference.py              # Translation inference pipeline
│   └── utils.py                  # Helpers: saving/loading, gradient accumulation
│
├── evaluation/
│   ├── metrics.py                # BLEU, METEOR, ROUGE-L, GLEU, BERTScore, BLEURT
│   ├── statistical_tests.py      # Wilcoxon, Friedman, post-hoc
│   ├── quantitative_analysis.py  # Metric tables & aggregation
│   └── visualization.py          # Graphs & plots
│
├── explainability/
│   ├── lime_analysis.py          # LIME token-level explanations
│   ├── shap_analysis.py          # SHAP token-level explanations
│   └── visualization.py          # Plots for LIME/SHAP
│
├── configs/
│   ├── config.yaml               # Dataset paths, training parameters, RLHF/SFT params
│   └── logging_config.yaml       # Logging setup
│
├── scripts/
│   ├── run_training.py           # CLI for training/fine-tuning
│   ├── run_inference.py          # CLI for translation inference
│   ├── run_evaluation.py         # CLI for metrics & statistical tests
│   └── run_explainability.py     # CLI for LIME/SHAP analysis
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
└── README.md                     # Project overview
```

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/TransLlamaX.git
cd TransLlamaX
```

2. Create a Python environment:

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## **Configuration**

* Update `configs/config.yaml` with:

  * Dataset paths (`raw` and `processed`)
  * Training hyperparameters (learning rate, batch size, LoRA/QLoRA ranks)
  * RLHF / preference optimization parameters
* Update `configs/logging_config.yaml` for custom logging setup.

---

## **Usage**

### **1. Training / Fine-Tuning**

```bash
python scripts/run_training.py --config configs/config.yaml
```

* Implements LoRA/QLoRA fine-tuning, supervised fine-tuning, and preference optimization (RLHF).
* Saves best checkpoints to `models/checkpoints/`.

### **2. Inference / Translation**

```bash
python scripts/run_inference.py --input "Kripya mere YouTube channel ko like aur subscribe karein"
```

* Returns English translation using the trained model.

### **3. Evaluation**

```bash
python scripts/run_evaluation.py --dataset data/processed/test.csv
```

* Calculates BLEU, METEOR, ROUGE-L, GLEU, BERTScore-F1, BLEURT
* Performs Wilcoxon signed-rank, Friedman, and post-hoc Conover–Holm tests
* Generates result tables and plots (figures 6–7).

### **4. Explainability**

```bash
python scripts/run_explainability.py --dataset data/processed/test.csv
```

* Generates token-level explanations using **LIME** and **SHAP**
* Produces plots showing semantic importance of tokens (figures 8–11)

---

## **Datasets**

* **FindNITAI public split**: benchmark Hinglish → English dataset
* **Custom creator-focused split**: informal, mixed-script social media text
* Both datasets are preprocessed and tokenized before training.

---

## **Results & Insights**

* **Quantitative**: TransLlamaX outperforms baselines across all metrics
* **Statistical**: Medium-to-large effect sizes, high probability-of-superiority, robust across splits
* **Qualitative**: LIME & SHAP confirm model relies on semantically central tokens (entities, actions, politeness), ignoring light function words
* Handles slang, romanized Hindi, and social media-specific terms effectively.

---

## **Dependencies**

* Python 3.10+
* PyTorch 2.1+
* HuggingFace Transformers, Accelerate
* Datasets, tokenizers
* BLEU, METEOR, ROUGE, GLEU, BERTScore, BLEURT
* LIME, SHAP, matplotlib, seaborn, pandas, numpy, scipy

---

_**Note:** This repository contains a sample dataset. To access the full custom dataset, you can refer to this repo: https://github.com/Shatakshi127/Hinglish_To_English_Dataset_
