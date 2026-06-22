# UK Road Traffic Accident Severity Prediction

This repository contains the code and outputs for my MSc Data Science project at the University of Hertfordshire.

## Project Overview

The aim of this project is to predict the severity of UK road traffic accidents (Fatal, Serious and Slight) using machine learning models while addressing the problem of class imbalance.

The project compares several machine learning algorithms together with different imbalance-handling techniques and evaluates their performance using macro F1-score, balanced accuracy and class-specific recall.

---

## Dataset

- UK STATS19 Road Traffic Accident Dataset
- Sample size: 60,000 records
- Target variable: **Accident_Severity**
    - Fatal
    - Serious
    - Slight

---

## Models

The following models were implemented:

- Logistic Regression
- Random Forest
- XGBoost
- Balanced Random Forest

Imbalance handling methods:

- Class Weighting
- RandomOverSampler
- SMOTENC
- Hyperparameter Tuning (RandomizedSearchCV)

---

## Best Model

**Random Forest + RandomOverSampler + Hyperparameter Tuning**

Performance:

| Metric | Score |
|--------|------:|
| Accuracy | 0.754 |
| Balanced Accuracy | 0.401 |
| Macro F1 | 0.372 |
| Weighted F1 | 0.762 |

---

## Project Structure

```
.
├── final_project_code.py
├── final_project_notebook.ipynb
├── Accident_Information.csv
├── README.md
└── project_outputs/
    ├── figures/
    ├── model_reports/
    ├── final_model_comparison_results_with_tuning.csv
    ├── tuned_random_forest_feature_importance.csv
    └── final_project_summary.txt
```

---

## Installation

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn xgboost
```

---

## Running the Project

```bash
python final_project_code.py
```

or open

```
final_project_notebook.ipynb
```

and run all cells.

---

## Author

**Rohit Bhatta**

MSc Data Science

University of Hertfordshire
