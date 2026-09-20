"""
streamlit_app.py
-----------------
Front end for the invoice classification system.

Lets a user upload a CSV of new invoices, runs them through the SAME
cleaning/feature-engineering pipeline used for training, predicts a
validation label for each with the saved model, logs the predictions
to the SQL database, and shows summary insights.

Run with:  streamlit run app/streamlit_app.py
(run from the project root so the relative imports/paths resolve)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # project root

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

from pipeline import clean_and_engineer
from database import InvoiceDB
from train_model import NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODEL_PATH

st.set_page_config(page_title="Invoice Intelligence", layout="wide")

st.title("🧾 Invoice Intelligence")
st.caption("ML-powered invoice classification — flags duplicates, amount anomalies, and missing-info invoices before they reach approval.")

db = InvoiceDB()

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train_model.py` first.")
    st.stop()

pipe = joblib.load(MODEL_PATH)

tab1, tab2, tab3 = st.tabs(["Classify New Invoices", "Vendor Insights", "Training Data Overview"])

# ---------------- TAB 1: classify uploaded invoices ----------------
with tab1:
    st.subheader("Upload invoices to classify")
    st.write("CSV columns expected: `invoice_id, vendor, category, invoice_date, due_date, payment_terms, amount, tax_amount`")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    sample = st.checkbox("Or use a sample of the existing raw data instead", value=not uploaded)

    if uploaded is not None:
        raw_new = pd.read_csv(uploaded)
    elif sample:
        raw_new = pd.read_csv("data/raw_invoices.csv").sample(15, random_state=7)
    else:
        raw_new = None

    if raw_new is not None:
        raw_new = raw_new.drop(columns=["label"], errors="ignore")  # unseen invoices don't have a true label
        processed = clean_and_engineer(raw_new.assign(label="Unknown"))

        X = processed[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        preds = pipe.predict(X)
        probs = pipe.predict_proba(X).max(axis=1)

        result = processed[["invoice_id", "vendor", "category", "amount"]].copy()
        result["predicted_label"] = preds
        result["confidence"] = (probs * 100).round(1).astype(str) + "%"

        def highlight(row):
            color = "" if row["predicted_label"] == "Valid" else "background-color: #ffe9e6"
            return [color] * len(row)

        st.dataframe(result.style.apply(highlight, axis=1), use_container_width=True)

        flagged = (result["predicted_label"] != "Valid").sum()
        st.info(f"{flagged} of {len(result)} invoices flagged for review.")

        log_df = pd.DataFrame({
            "invoice_id": result["invoice_id"],
            "predicted_label": result["predicted_label"],
            "confidence": probs,
            "predicted_at": datetime.now().isoformat(timespec="seconds"),
        })
        if st.button("Log these predictions to the database"):
            db.log_predictions(log_df)
            st.success("Predictions logged.")

# ---------------- TAB 2: vendor-level insights (SQL aggregation) ----------------
with tab2:
    st.subheader("Vendor summary (computed in SQL)")
    vendor_df = db.vendor_summary()
    st.dataframe(vendor_df, use_container_width=True)
    st.bar_chart(vendor_df.set_index("vendor")["flagged_count"])

# ---------------- TAB 3: training data overview ----------------
with tab3:
    st.subheader("Label distribution in training data")
    processed_all = db.get_processed()
    st.bar_chart(processed_all["label"].value_counts())
    st.subheader("Sample of processed/feature-engineered data")
    st.dataframe(processed_all.head(20), use_container_width=True)
