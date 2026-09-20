"""
train_model.py
---------------
Trains a classifier to predict an invoice's validation label:
    Valid / Duplicate / Amount Anomaly / Missing Info

Model choice: RandomForestClassifier.
  - Handles a mix of numeric + categorical-encoded features well
  - Robust to outliers/scale differences, so little preprocessing needed
  - Gives feature importances, useful to explain *why* an invoice was flagged
  - class_weight="balanced" because "Valid" invoices heavily outnumber
    flagged ones (this is realistic -- most invoices ARE fine)

Evaluation:
  - Stratified train/test split so rare classes appear in both sets
  - Precision/recall/F1 per class (accuracy alone is misleading on
    imbalanced data -- a model that always predicts "Valid" would still
    score ~80% accuracy while catching zero problem invoices)
  - Confusion matrix to see which flagged types get confused with which
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from database import InvoiceDB

MODEL_PATH = Path("models/invoice_classifier.joblib")

NUMERIC_FEATURES = [
    "amount", "tax_amount", "days_to_due", "invoice_month", "invoice_weekday",
    "vendor_avg_amount", "amount_zscore_vendor",
    "missing_field_count", "is_potential_duplicate",
]
CATEGORICAL_FEATURES = ["category"]  # vendor excluded: too high-cardinality
                                       # for one-hot; amount_zscore_vendor
                                       # already encodes vendor-relative info


def build_pipeline():
    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def main():
    db = InvoiceDB()
    df = db.get_processed()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                          columns=[f"pred_{l}" for l in labels])
    print(cm_df)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nMacro F1: {macro_f1:.3f}")

    # feature importances (only meaningful for the numeric block + one-hot cats)
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    feature_names = NUMERIC_FEATURES + list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    importances = pipe.named_steps["clf"].feature_importances_
    imp_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\n=== Top Feature Importances ===")
    print(imp_df.head(8))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
