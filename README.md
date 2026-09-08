# Fraud Detection — AI Solutions Engineering Hackathon

## Problem Statement
Fraudulent transactions cost financial institutions and customers billions annually.
This project builds a machine learning model to automatically flag potentially
fraudulent transactions, mirroring the kind of fraud detection needed in digital
wallets and banking platforms like EasyPaisa/JazzCash.

## Dataset
- **Source:** Credit Card Fraud Detection (Kaggle, ULB/Worldline)
- **Size:** 284,807 transactions, 492 confirmed fraud cases (~0.17%)
- **Features:** `Time`, `Amount`, and 28 anonymized PCA features (`V1`–`V28`)

## Approach
1. **EDA & Cleaning** — checked for missing values/duplicates, analyzed class
   imbalance and transaction amount distributions, identified top fraud-correlated
   features via a correlation heatmap.
2. **Feature Engineering** — created 5 new features: scaled Amount/Time,
   `Transaction_Hour`, two interaction features (`V17_V14_Interaction`,
   `V12_V10_Interaction`) built from the most fraud-correlated variables, and a
   binned `Amount_Category`.
3. **Class Imbalance Handling** — undersampled the majority (legit) class to a
   5:1 ratio to prevent the model from ignoring the rare fraud class.
4. **Modeling** — trained a `RandomForestClassifier` (100 trees) on an 80/20
   train-test split.
5. **Evaluation** — assessed performance using Accuracy, Precision, Recall,
   F1-score, and a Confusion Matrix (not accuracy alone, given the imbalance).

## Results
| Metric | Value |
|---|---|
| Accuracy | 96.65% |
| Precision (Fraud) | 0.95 |
| Recall (Fraud) | 0.84 |
| F1-score (Fraud) | 0.89 |

**Confusion Matrix:**
- 469 legit correctly identified, 4 false alarms
- 80 fraud cases correctly caught, 15 missed

The two engineered interaction features (`V12_V10_Interaction`, `V17_V14_Interaction`)
ranked among the top predictors, validating our feature engineering choices.

## Future Improvements
- Try SMOTE oversampling instead of undersampling to retain more legit-transaction data
- Experiment with an MLPClassifier (neural network) for comparison
- Tune the classification threshold to improve fraud recall further

## Team & Contributions
| Member | Contribution |
|---|---|
| Qaiser | Problem definition, data ingestion |
| Aliyya | EDA, data cleaning |
| Qaiser| Feature engineering |
| Naiba Batool | Model training, evaluation, GitHub workflow, documentation |

## How to Run
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Open `fraud_detection_model.ipynb` in Google Colab or Jupyter
3. Update the file path to your dataset location
4. Run all cells in order
