"""
Baselines for the TabPFN experiments, run on the exact same splits as the notebooks.
Run from the repo root:  python src/baselines.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def to_categories(X):
    """Gradient boosting needs text columns marked as categories."""
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category")
    return X


def gbm_classifier():
    return HistGradientBoostingClassifier(random_state=0, categorical_features="from_dtype")


# ---------------- Richter: earthquake building damage (micro-F1) ----------------
X = pd.read_csv("datasets/richter_dataset/train_values.csv").drop(columns=["building_id"])
y = pd.read_csv("datasets/richter_dataset/train_labels.csv").drop(columns=["building_id"]).iloc[:, 0]
X = to_categories(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, y_val = X_val.iloc[:500], y_val.iloc[:500]

small = gbm_classifier().fit(X_train.iloc[:10000], y_train.iloc[:10000])
full = gbm_classifier().fit(X_train, y_train)
print("Richter | GBM on 10,000 rows:", f1_score(y_val, small.predict(X_val), average="micro"))
print(f"Richter | GBM on {len(X_train):,} rows:", f1_score(y_val, full.predict(X_val), average="micro"))
print("Richter | always predict most common class:", (y_val == y_train.mode()[0]).mean())

# ---------------- Flu shot: vaccine uptake (ROC-AUC) ----------------
X = to_categories(pd.read_csv("datasets/flushot_dataset/training_set_features.csv").drop(columns=["respondent_id"]))
labels = pd.read_csv("datasets/flushot_dataset/training_set_labels.csv")
for target in ["h1n1_vaccine", "seasonal_vaccine"]:
    y = labels[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, y_val = X_val.iloc[:500], y_val.iloc[:500]
    small = gbm_classifier().fit(X_train.iloc[:3000], y_train.iloc[:3000])
    full = gbm_classifier().fit(X_train, y_train)
    print(f"{target} | GBM on 3,000 rows:", roc_auc_score(y_val, small.predict_proba(X_val)[:, 1]))
    print(f"{target} | GBM on {len(X_train):,} rows:", roc_auc_score(y_val, full.predict_proba(X_val)[:, 1]))

# ---------------- MSEDCL: monthly SAIFI per circle (MAE, WAPE) ----------------
target = "saifi"
df = pd.read_csv("datasets/msedcl_reliability_dataset/msedcl_reliability.csv")
df = df.dropna(subset=[target])
df["month"] = pd.to_datetime(df["report_month"]).dt.month
df["circle_id"] = df["circle"].astype("category").cat.codes
df["value_3m_ago"] = df.groupby("circle")[target].shift(3)
df["value_1y_ago"] = df.groupby("circle")[target].shift(12)
df = df.dropna(subset=["value_3m_ago", "value_1y_ago"]).sort_values("report_month")

features = ["circle_id", "month", "value_3m_ago", "value_1y_ago"]
X_train, X_val, y_train, y_val = train_test_split(df[features], df[target], test_size=500, shuffle=False)
X_train, y_train = X_train.iloc[-3000:], y_train.iloc[-3000:]


def report(name, preds):
    wape = np.abs(y_val - preds).sum() / y_val.sum() * 100
    print(f"SAIFI | {name}: MAE {mean_absolute_error(y_val, preds):.3f}, WAPE {wape:.1f}%")


report("GBM", HistGradientBoostingRegressor(random_state=0).fit(X_train, y_train).predict(X_val))
report("repeat value from 3 months ago", X_val["value_3m_ago"])
report("repeat value from 1 year ago", X_val["value_1y_ago"])
