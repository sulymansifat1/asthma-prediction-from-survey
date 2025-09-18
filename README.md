# Asthma Prediction from Survey — ML Lab Project

An end-to-end mini lab project to predict asthma risk from real survey responses using classical machine learning. The workflow covers data cleaning, exploratory analysis, model comparison, imbalance handling (SMOTE), threshold tuning, and a simple deployment as an interactive quiz.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" />
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white" />
  <img alt="pandas" src="https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white" />
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white" />
  <img alt="Seaborn" src="https://img.shields.io/badge/Seaborn-0.13.x-4C9A2A" />
  <img alt="imbalanced-learn" src="https://img.shields.io/badge/imblearn-SMOTE-8A2BE2" />
  <img alt="Status" src="https://img.shields.io/badge/status-student%20project-blue" />
  <img alt="License" src="https://img.shields.io/badge/license-TBD-lightgrey" />
  <img alt="Platform" src="https://img.shields.io/badge/platform-Colab%20%7C%20Hugging%20Face%20Spaces-yellow" />
  
</p>

> Course: Data Mining & Machine Learning Lab (CSE322) • Department of CSE, DIU
>
> Live demo (quiz): https://huggingface.co/spaces/Sulymansifat/asthma-prediction-quiz
>
> Notebook (Colab): https://colab.research.google.com/drive/1VxENFQY3UaJ82u80CpLvVs3R-nvcxksF?usp=sharing
>
> Report: `DMML Lab Final Project Report.pdf`
>
> Supervisor: Mr. Md. Abdullah Al Kafi, Lecturer, Dept. of CSE, Daffodil International University

---

## Table of Contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
  - [Run in Colab](#run-in-colab)
  - [Run locally (VS Code/Jupyter)](#run-locally-vs-codejupyter)
- [Usage](#usage)
- [Methods](#methods)
- [Results](#results)
- [Roadmap](#roadmap)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview
This project predicts whether a person is an asthma patient from a real Google Forms survey. It follows a classic ML pipeline: data preparation, exploratory analysis, baseline models, ensemble methods, handling class imbalance, model selection, and lightweight deployment as a quiz on Hugging Face Spaces.

- Data source: a custom survey with demographics, awareness, respiratory symptoms, and environmental exposure.
- Best performing model: Random Forest (chosen for balanced accuracy/recall and robustness).
- Deployment: an interactive quiz experience that mirrors the trained model’s feature set.

Note: This is an educational project; it does not provide medical advice and should not be used for clinical decision-making.

## Highlights
- Real-world responses collected via Google Forms; included as `Asthma Data Collection Survey.csv`
- Consistent preprocessing (column normalization, label encoding)
- Multiple models evaluated: Logistic Regression, SVM, Decision Tree, Random Forest, Naive Bayes
- Imbalance handling with SMOTE and decision threshold exploration
- Confusion-matrix visualizations and classification reports per model
- Deployed interactive quiz UI on Hugging Face Spaces

## Tech Stack
- Language: Python (3.8+)
- Core libraries: scikit-learn, pandas, numpy, matplotlib, seaborn, imbalanced-learn, joblib
- Platforms/Tools: Google Colab, VS Code Jupyter, Hugging Face Spaces

## Dataset
File: `Asthma Data Collection Survey.csv`

Columns (from header):
- Gender
- Age
- Living area
- Pollution
- aware of asthma?
- asthma patient?  ← target label (Yes/No)
- allergy?
- coughing or throat problem ?
- breathing problem? 
- smoke or tobacco user? 
- any other health issues?

Key notes:
- Raw headers include spaces, mixed case, and question marks. In the notebook, columns are normalized to lowercase with underscores for modeling.
- Categorical variables are label-encoded. Keep encoders consistent between training and inference.
- Some responses are imbalanced; SMOTE is used to upsample the minority class during training.

## Folder Structure
This repository is intentionally simple and notebook-centric:

```
.
├─ Asthma Data Collection Survey.csv     # Survey data (raw headers)
├─ DMML Lab Final Project Report.pdf     # Final write-up (PDF)
└─ Prediction of Asthma.ipynb            # Main analysis & modeling notebook
```

## Getting Started

### Run in Colab
1. Open the Colab link above.
2. Upload `Asthma Data Collection Survey.csv` to your Colab runtime or mount Google Drive and update `file_path` accordingly.
3. Run all cells to reproduce preprocessing, model training, evaluation, and (optional) model export with joblib.

### Run locally (VS Code/Jupyter)
1. Install Python 3.8+ and VS Code with the Jupyter extension.
2. Create a virtual environment and install common packages (scikit-learn, pandas, numpy, seaborn, matplotlib, imbalanced-learn, joblib).
3. Open `Prediction of Asthma.ipynb` and execute cells. Ensure the dataset path points to `Asthma Data Collection Survey.csv` in the repo root.

Tips:
- If you export a trained model (e.g., `optimized_asthma_model.pkl`), also save the preprocessing steps (encoders) or wrap them in a scikit-learn `Pipeline` for reliable inference.

## Usage
- Web UI (no code): Use the interactive quiz here: https://huggingface.co/spaces/Sulymansifat/asthma-prediction-quiz
- Notebook: Explore EDA, model comparison, SMOTE, threshold tuning, and confusion matrices.
- Optional local inference: If you saved a model with its preprocessing, load it with `joblib` and pass a single-row dataframe with the same feature schema used for training.

## Methods
Preprocessing
- Column normalization: strip/replace spaces with underscores and lowercase names
- Encoding: label-encode categorical variables
- Split: train/test (commonly 80/20 in this notebook)

Modeling
- Baselines: Logistic Regression, Naive Bayes
- Nonlinear models: SVM, Decision Tree
- Ensemble: Random Forest (selected)
- Hyperparameter search: `GridSearchCV` (for Random Forest)
- Class imbalance: `SMOTE` (imbalanced-learn)
- Threshold tuning: explore precision–recall trade-offs for medical-context sensitivity

Evaluation
- Metrics: accuracy, precision, recall, F1-score
- Diagnostics: confusion matrix heatmaps per model
- Selection: prioritize minimizing false negatives while maintaining balanced performance

Artifacts
- Optional export: save the best model with `joblib.dump(...)`
- Recommended: persist a full `Pipeline` (preprocessing + model) for robust deployment

## Results
This section summarizes the actual outputs printed in the notebook. Scores vary by split and random seed; see notes below for context.

Test split A (small test set of 10; heavily imbalanced):
- Logistic Regression — Accuracy: 0.80; positive-class recall: 0.00
- SVM — Accuracy: 0.90; positive-class recall: 0.00
- Decision Tree — Accuracy: 0.90; positive-class recall: 0.00
- Random Forest — Accuracy: 0.90; positive-class recall: 0.00

Model selection (Random Forest hyperparameters via GridSearchCV):
- Best CV accuracy: 0.925
- Best params: { n_estimators: 50, max_depth: None, min_samples_split: 5, min_samples_leaf: 1 }
- Optimized RF test accuracy on split A: 0.90

Test split B (example with both classes well represented, Random Forest):
- Accuracy: 0.89
- Confusion Matrix: [[10, 1], [1, 7]] (N = 19)
- Class-wise metrics:
  - Class 0 — precision: 0.91, recall: 0.91, f1: 0.91
  - Class 1 — precision: 0.88, recall: 0.88, f1: 0.88

SMOTE and threshold tuning:
- SMOTE improved the minority-class learnability in training; threshold experiments (e.g., τ = 0.3) show how to trade precision for recall—important in screening to reduce false negatives.

Takeaways:
- Random Forest is the most balanced choice across runs on this dataset.
- Very small or imbalanced test splits can inflate accuracy while missing positives. Prefer stratified splits, cross-validation, and monitoring recall for the positive class.
- Results will change with data size, split strategy, and seeds; re-run the notebook to reproduce your exact environment.

## Roadmap
- Save and version a full scikit-learn `Pipeline` (encoders + model)
- Add a small inference script and example inputs (CLI/Streamlit)
- Add cross-validation and confidence intervals for metrics
- Improve explainability (feature importance, SHAP)
- Expand features (air quality indices, longitudinal symptoms)
- Package requirements and an environment file
- Add a license file

## Authors
- Md. Sulyman Islam Sifat — ID: 211-15-4004
- Mohammed Nazmul Hoque Shawon — ID: 211-15-3996

Supervisor: Mr. Md. Abdullah Al Kafi — Lecturer, Dept. of CSE, DIU

## License
No license specified yet. Consider adding a `LICENSE` file (e.g., MIT) to clarify reuse.

## Acknowledgments
- Daffodil International University (DIU) — CSE322 Data Mining & Machine Learning Lab
- Survey participants for providing real-world responses
- scikit-learn, pandas, numpy, seaborn, imbalanced-learn
- Hugging Face Spaces and Google Colab for accessible experimentation and deployment
