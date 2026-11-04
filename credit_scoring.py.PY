"""
Credit scoring model: predict the probability that a borrower will default.

What it does
  1. Loads your CSV (--data file.csv --target default) or, with no file, generates a realistic synthetic dataset.
  2. Feature engineering: debt-to-income, loan-to-income, payment-to-income, late payments per credit line,
     risk flags and age groups (printed at run time and saved to engineered_features.csv)
  3. Trains Logistic Regression, Decision Tree and Random Forest (with imputation, scaling, one-hot encoding).
  4. Evaluates each on a held-out test set: Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrix,
     plus 5-fold cross-validated ROC-AUC.
  5. Saves the best model, comparison table and plots, and converts default probability to a 300-850 credit score.

Usage
  python credit_scoring.py                                   # synthetic data demo
  python credit_scoring.py --data loans.csv --target default # your data (target: 1 = defaulted / bad, 0 = good)
  python credit_scoring.py --predict credit_output/sample_applicants.json   # score new applicants

Requires: pandas numpy scikit-learn matplotlib joblib
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ----------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------- #
def make_synthetic_data(n=6000, seed=42, default_rate=0.18):
    """Synthetic borrower data where default risk depends on realistic factors plus noise."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    annual_income = np.round(np.exp(rng.normal(10.9, 0.5, n)), -2)
    employment_years = np.round(np.minimum(rng.gamma(2.0, 3.0, n), age - 18), 1)
    credit_history_years = np.round(np.clip(rng.gamma(3.0, 3.5, n), 0.5, age - 18), 1)
    num_credit_lines = rng.poisson(5, n) + 1
    credit_utilization = np.round(np.clip(rng.beta(2, 4, n), 0, 1), 3)
    monthly_debt_payments = np.round(annual_income / 12 * np.clip(rng.beta(2, 6, n) * 1.2, 0, 0.9), 0)
    loan_amount = np.round(annual_income * rng.uniform(0.05, 0.8, n), -2)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n)
    late_payments_12m = rng.poisson(0.3 + 2.5 * credit_utilization ** 2)
    num_inquiries_6m = rng.poisson(1.2, n)
    home_ownership = rng.choice(["rent", "mortgage", "own"], n, p=[0.4, 0.4, 0.2])
    loan_purpose = rng.choice(["debt_consolidation", "home_improvement", "car", "education", "medical", "other"], n)

    dti = monthly_debt_payments * 12 / annual_income
    lti = loan_amount / annual_income
    z = (3.0 * dti + 1.5 * lti + 2.2 * credit_utilization + 0.45 * late_payments_12m
         + 0.15 * num_inquiries_6m - 0.05 * credit_history_years - 0.06 * employment_years
         + 0.3 * (home_ownership == "rent") - 0.3 * (home_ownership == "own") + rng.normal(0, 0.5, n))
    lo, hi = -20.0, 20.0  # find the intercept that gives the requested default rate
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > default_rate:
            hi = mid
        else:
            lo = mid
    p = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))

    df = pd.DataFrame({
        "age": age, "annual_income": annual_income, "employment_years": employment_years,
        "credit_history_years": credit_history_years, "num_credit_lines": num_credit_lines,
        "credit_utilization": credit_utilization, "monthly_debt_payments": monthly_debt_payments,
        "loan_amount": loan_amount, "loan_term_months": loan_term_months,
        "late_payments_12m": late_payments_12m, "num_inquiries_6m": num_inquiries_6m,
        "home_ownership": home_ownership, "loan_purpose": loan_purpose,
        "default": (rng.random(n) < p).astype(int),
    })
    for col in ["employment_years", "credit_utilization"]:  # a few missing values, like real data
        df.loc[rng.random(n) < 0.03, col] = np.nan
    return df


FEATURE_DOCS = {
    "debt_to_income": "monthly_debt_payments * 12 / annual_income",
    "loan_to_income": "loan_amount / annual_income",
    "monthly_loan_payment": "loan_amount / loan_term_months",
    "payment_to_income": "monthly_loan_payment / (annual_income / 12)",
    "late_per_line": "late_payments_12m / num_credit_lines",
    "avg_account_age": "credit_history_years / num_credit_lines",
    "log_income": "log(1 + annual_income)",
    "high_utilization": "1 if credit_utilization > 0.7",
    "risk_interaction": "credit_utilization * (1 + late_payments_12m)",
    "debt_burden_high": "1 if debt_to_income > 0.4",
    "short_employment": "1 if employment_years < 1",
    "many_inquiries": "1 if num_inquiries_6m >= 3",
    "age_group": "age bucket: <=25, 26-35, 36-45, 46-55, 56+",
}


def engineer_features(df):
    """Create ratio and risk features from financial history (only when the needed columns exist)."""
    d = df.copy()
    has = lambda *cols: all(c in d.columns for c in cols)
    if has("monthly_debt_payments", "annual_income"):
        d["debt_to_income"] = d["monthly_debt_payments"] * 12 / d["annual_income"]
    if has("loan_amount", "annual_income"):
        d["loan_to_income"] = d["loan_amount"] / d["annual_income"]
    if has("loan_amount", "loan_term_months"):
        d["monthly_loan_payment"] = d["loan_amount"] / d["loan_term_months"]
        if has("annual_income"):
            d["payment_to_income"] = d["monthly_loan_payment"] / (d["annual_income"] / 12)
    if has("late_payments_12m", "num_credit_lines"):
        d["late_per_line"] = d["late_payments_12m"] / d["num_credit_lines"]
    if has("credit_history_years", "num_credit_lines"):
        d["avg_account_age"] = d["credit_history_years"] / d["num_credit_lines"]
    if has("annual_income"):
        d["log_income"] = np.log1p(d["annual_income"].clip(lower=0))
    if has("credit_utilization"):
        d["high_utilization"] = (d["credit_utilization"] > 0.7).astype(float).where(d["credit_utilization"].notna())
    if has("credit_utilization", "late_payments_12m"):
        d["risk_interaction"] = d["credit_utilization"] * (1 + d["late_payments_12m"])
    if "debt_to_income" in d.columns:
        d["debt_burden_high"] = (d["debt_to_income"] > 0.4).astype(float).where(d["debt_to_income"].notna())
    if has("employment_years"):
        d["short_employment"] = (d["employment_years"] < 1).astype(float).where(d["employment_years"].notna())
    if has("num_inquiries_6m"):
        d["many_inquiries"] = (d["num_inquiries_6m"] >= 3).astype(float).where(d["num_inquiries_6m"].notna())
    if has("age"):
        d["age_group"] = pd.cut(d["age"], [0, 25, 35, 45, 55, 200],
                                labels=["<=25", "26-35", "36-45", "46-55", "56+"]).astype(str)
    return d.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------- #
# Models
# ----------------------------------------------------------------------------- #
def make_pipelines(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    def prep(scale):
        num_steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
        return ColumnTransformer([
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ])

    return {
        "Logistic Regression": Pipeline([("prep", prep(True)),
                                         ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Decision Tree": Pipeline([("prep", prep(False)),
                                   ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=50,
                                                                    class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline([("prep", prep(False)),
                                   ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                                    class_weight="balanced", n_jobs=-1,
                                                                    random_state=42))]),
    }


def to_credit_score(p_default):
    """Map probability of default to a 300-850 score (higher = more creditworthy)."""
    return np.round(300 + (1 - np.asarray(p_default)) * 550).astype(int)


def risk_band(score):
    return "Low risk" if score >= 740 else "Moderate risk" if score >= 670 else "High risk" if score >= 580 else "Very high risk"


# ----------------------------------------------------------------------------- #
# Train / evaluate
# ----------------------------------------------------------------------------- #
def train_and_evaluate(df, target, out, test_size, threshold, seed):
    y = df[target].astype(int)
    if set(y.unique()) - {0, 1}:
        raise SystemExit(f"Target column '{target}' must contain only 0 (good) and 1 (bad/default).")
    raw_cols = [c for c in df.columns if c != target]
    X = engineer_features(df.drop(columns=[target]))
    new_cols = [c for c in X.columns if c not in raw_cols]
    print(f"Rows: {len(X)}, default rate: {y.mean():.1%}")
    print(f"\nFEATURE ENGINEERING: {len(raw_cols)} original features -> {X.shape[1]} features ({len(new_cols)} new)")
    for c in new_cols:
        print(f"  {c:18s} = {FEATURE_DOCS.get(c, '')}")
    pd.concat([X, y.rename(target)], axis=1).to_csv(out / "engineered_features.csv", index=False)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    pipes = make_pipelines(X)

    rows, probs, fitted = [], {}, {}
    for name, pipe in pipes.items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc").mean()
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        pred = (p >= threshold).astype(int)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, zero_division=0),
            "Recall": recall_score(y_te, pred),
            "F1-Score": f1_score(y_te, pred),
            "ROC-AUC": roc_auc_score(y_te, p),
            "CV ROC-AUC": cv_auc,
        })
        probs[name], fitted[name] = p, pipe

    res = pd.DataFrame(rows).set_index("Model").sort_values("ROC-AUC", ascending=False)
    print(f"\nTest-set results (default = positive class, threshold {threshold}):")
    print(res.round(3).to_string())
    res.round(4).to_csv(out / "model_comparison.csv")

    best = res.index[0]
    cm = confusion_matrix(y_te, (probs[best] >= threshold).astype(int))
    print(f"\nBest model by ROC-AUC: {best}")
    print(f"Confusion matrix (rows = actual good/bad, columns = predicted good/bad):\n{cm}")

    # Plots
    plt.figure(figsize=(6, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(out / "roc_curves.png", dpi=150); plt.close()

    rf = fitted["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values().tail(12)
    plt.figure(figsize=(7, 5))
    imp.plot.barh(color="#0B6E4F")
    plt.title("Random Forest: top 12 features"); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(out / "feature_importance.png", dpi=150); plt.close()

    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Greens")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, v, ha="center", va="center")
    plt.xticks([0, 1], ["Good", "Bad"]); plt.yticks([0, 1], ["Good", "Bad"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{best}: confusion matrix")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()

    joblib.dump({"model": fitted[best], "name": best, "features": X.columns.tolist(),
                 "raw_features": [c for c in df.columns if c != target], "threshold": threshold},
                out / "credit_model.joblib")
    print(f"\nSaved model, metrics table and plots to {out}/")
    return df.drop(columns=[target]).iloc[:3], best


# ----------------------------------------------------------------------------- #
# Scoring new applicants
# ----------------------------------------------------------------------------- #
def score_applicants(bundle, applicants):
    d = pd.DataFrame(applicants)
    d = d.reindex(columns=bundle["raw_features"])
    X = engineer_features(d)[bundle["features"]]
    p = bundle["model"].predict_proba(X)[:, 1]
    scores = to_credit_score(p)
    return pd.DataFrame({
        "default_probability": np.round(p, 3),
        "credit_score": scores,
        "risk_band": [risk_band(s) for s in scores],
        "decision": np.where(p >= bundle["threshold"], "Decline / review", "Approve"),
    })


def main():
    ap = argparse.ArgumentParser(description="Credit scoring model")
    ap.add_argument("--data", help="CSV file with borrower data (omit to use synthetic data)")
    ap.add_argument("--target", default="default", help="name of the 0/1 target column (1 = default)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5, help="default-probability cut-off for 'bad'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="credit_output")
    ap.add_argument("--predict", help="JSON file with one applicant (object) or several (list) to score")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)

    if a.predict:
        model_path = out / "credit_model.joblib"
        if not model_path.exists():
            raise SystemExit("Train first: python credit_scoring.py")
        data = json.loads(Path(a.predict).read_text())
        data = [data] if isinstance(data, dict) else data
        print(score_applicants(joblib.load(model_path), data).to_string())
        return

    if a.data:
        df = pd.read_csv(a.data)
        if a.target not in df.columns:
            raise SystemExit(f"Target column '{a.target}' not found. Columns: {list(df.columns)}")
    else:
        print("No --data given: using a synthetic dataset.")
        df = make_synthetic_data(seed=a.seed)
        df.to_csv(out / "synthetic_credit_data.csv", index=False)

    sample, best = train_and_evaluate(df, a.target, out, a.test_size, a.threshold, a.seed)

    sample_path = out / "sample_applicants.json"
    sample_path.write_text(json.dumps(json.loads(sample.to_json(orient="records")), indent=2))
    print(f"\nExample scores for 3 applicants ({sample_path}):")
    print(score_applicants(joblib.load(out / "credit_model.joblib"), json.loads(sample_path.read_text())).to_string()).


if __name__ == "__main__":
    main()
