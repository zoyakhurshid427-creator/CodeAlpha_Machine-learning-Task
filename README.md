# Credit Scoring Model

## Project Overview
This project predicts the probability of loan default using machine learning classification algorithms.

## Features
- Synthetic borrower dataset generation
- Financial feature engineering
- Logistic Regression
- Decision Tree
- Random Forest
- Model evaluation using Accuracy, Precision, Recall, F1-Score and ROC-AUC
- Confusion matrix and ROC curve visualization
- Credit score estimation and risk classification

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Project

Run with synthetic data:

```bash
python credit_scoring.py
```

Run with your own CSV dataset:

```bash
python credit_scoring.py --data loans.csv --target default
```

## Output
The project generates:
- Trained machine learning model
- Model comparison results
- Engineered features CSV
- ROC curves
- Feature importance plot
- Confusion matrix
- Sample applicant predictions

## Dataset Note
The default demonstration uses a synthetic dataset. Results are for educational purposes and do not represent validated real-world credit assessments.

## Author
Your Name