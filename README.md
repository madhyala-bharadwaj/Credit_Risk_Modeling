
# Credit Risk Modelling Project

## Overview

This project focuses on building credit risk models using various machine learning algorithms. The process includes data cleaning, feature selection, model training, evaluation, hyperparameter tuning and predicting on unseen data.


## Directory Structure

```
CreditRiskProject/
├── data/
│ ├── case_study1.xlsx
│ ├── case_study2.xlsx
│ └── Unseen_Dataset.xlsx
├── logs/
│ └── credit_risk_logs.log
├── credit_risk.ipynb
├── credit_risk.py
├── tuning_results.xlsx
├── final_predictions.xlsx
├── README.md
└── .gitignore
```


## Requirements

- pandas
- numpy
- scikit-learn
- xgboost
- statsmodels
- scipy
- logging
- joblib


## Files

- `case_study1.xlsx` and `case_study2.xlsx`: Input datasets used for model training.
- `Unseen_Dataset.xlsx`: Dataset containing unseen data for making predictions.
- `tuning_results.xlsx`: Excel file containing results from hyperparameter tuning.
- `final_predictions.xlsx`: Excel file with final predictions on the unseen dataset.
- `credit_risk.py`: Python script version of the credit risk modeling process.
- `credit_risk.ipynb`: Jupyter Notebook version with detailed implementation and analysis.
- `README.md`: This documentation file providing an overview of the project.


### Project Steps

### 1. Data Reading and Cleaning

The project starts by reading data from `case_study1.xlsx` and `case_study2.xlsx`. Initial cleaning involves identifying and handling missing values, where features with more than 20% missing values are removed entirely. Additionally, rows containing missing values are dropped due to the dataset's large size.

### 2. Feature Selection

1. **Chi-Squared Test**: Categorical columns are analyzed using the Chi-squared test to determine their significance in relation to the target variable (`Approved_Flag`). Features that do not show significant association are removed from the dataset.
2. **Multicollinearity Check (Variance Inflation Factor)**: Numeric columns are checked for multicollinearity using the Variance Inflation Factor (VIF). Features with high VIF values (greater than 6) are removed to enhance model performance and interpretability.
3. **ANOVA Test**: An ANOVA test is conducted on numeric features to identify those that significantly differ across different classes of the target variable (`P1`, `P2`, `P3`, `P4`). Features failing to show significant differences are excluded from further analysis.

### 3. Encoding

Education levels are mapped to numeric values for consistency (`SSC`, `12TH`, `GRADUATE`, `UNDER GRADUATE`, `POST-GRADUATE`, `OTHERS`, `PROFESSIONAL`). Categorical features are then one-hot encoded to prepare the data for modeling.

### 4. Scaling

Selected numeric features undergo scaling using `StandardScaler` to normalize their values and improve model convergence during training.

### 5. Model Training and Evaluation

Model Selection
Several machine learning models are trained and evaluated using the processed data:

- XGBoost
- Decision Tree
- Random Forest
- Logistic Regression
- SVM
- KNN
- Gradient Boosting
- Naive Bayes
- AdaBoost
- Extra Trees
Each model is evaluated based on metrics such as accuracy, precision, recall, and F1 score to determine its performance in predicting the target variable classes (`P1`, `P2`, `P3`, `P4`).

### 6. Hyperparameter Tuning
The best performing model(`XGBoost`) undergo hyperparameter tuning using `GridSearchCV`. Parameters are optimized to further enhance model accuracy and generalization on unseen data. Tuning results are logged and saved in `tuning_results.xlsx`.

### 7. Predicting on Unseen Data

The final trained model (`XGBoost Classifier`) is applied to `Unseen_Dataset.xlsx` for making predictions on the target variable. Predictions are saved in `final_predictions.xlsx` along with other relevant data columns.


## Logging

The project includes logging functionality to track progress, errors, and information during the execution of the script. Logs provide insights into model training, evaluation, and prediction stages.
Logs are saved in the `logs/credit_risk_logs.log` directory.

## Error Handling

The script includes try-except blocks to handle potential errors gracefully, maintaining smooth execution and preventing script crashes.
