"""
pipeline.py
-----------
Data cleaning + feature engineering.

Takes raw invoices (as they'd arrive from an upload/extraction tool -- messy,
some missing fields, some duplicates) and turns them into a clean feature
table the model can be trained on. This is the step that decides *what
signals* the model is actually allowed to learn from.

Engineered features:
    days_to_due            -- payment window length (a very late/short window
                               is unusual)
    invoice_month/weekday   -- seasonality / submission-pattern signals
    vendor_avg_amount       -- what "normal" looks like for this vendor
    amount_zscore_vendor    -- how far this invoice's amount is from that
                               vendor's normal, in standard deviations
                               (this is the main signal for Amount Anomaly)
    missing_field_count     -- how many required fields are blank
                               (main signal for Missing Info)
    is_potential_duplicate  -- same vendor + amount, invoice dates within
                               5 days of each other (main signal for Duplicate)
"""

import pandas as pd
import numpy as np


REQUIRED_FIELDS = ["vendor", "category", "amount", "due_date"]


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- missing-field count (compute BEFORE filling anything in) ---
    df["missing_field_count"] = df[REQUIRED_FIELDS].isna().sum(axis=1)

    # --- fill/clean for downstream calculations, without destroying the
    #     "missing info" signal we already captured above ---
    df["vendor"] = df["vendor"].fillna("Unknown Vendor")
    df["category"] = df["category"].fillna("Uncategorized")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["amount"] = df["amount"].fillna(df.groupby("vendor")["amount"].transform("median"))
    df["amount"] = df["amount"].fillna(df["amount"].median())

    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
    # if due_date missing, estimate from median payment window for that vendor
    median_window = (df["due_date"] - df["invoice_date"]).dt.days.median()
    if pd.isna(median_window):
        median_window = 30
    est_due = df["invoice_date"] + pd.to_timedelta(median_window, unit="D")
    df["due_date"] = df["due_date"].fillna(est_due)

    df["days_to_due"] = (df["due_date"] - df["invoice_date"]).dt.days
    df["invoice_month"] = df["invoice_date"].dt.month
    df["invoice_weekday"] = df["invoice_date"].dt.weekday

    # --- vendor-level amount profile ---
    vendor_stats = df.groupby("vendor")["amount"].agg(["mean", "std"]).rename(
        columns={"mean": "vendor_avg_amount", "std": "vendor_std_amount"}
    )
    df = df.merge(vendor_stats, on="vendor", how="left")
    df["vendor_std_amount"] = df["vendor_std_amount"].replace(0, np.nan).fillna(df["amount"].std())
    df["amount_zscore_vendor"] = (
        (df["amount"] - df["vendor_avg_amount"]) / df["vendor_std_amount"]
    ).fillna(0)

    # --- duplicate detection: same vendor + amount (rounded), dates close together ---
    df = df.sort_values(["vendor", "amount", "invoice_date"])
    df["is_potential_duplicate"] = 0
    for _, group in df.groupby(["vendor", df["amount"].round(2)]):
        if len(group) < 2:
            continue
        dates = group["invoice_date"].sort_values()
        gaps = dates.diff().dt.days
        close_pairs = gaps[gaps <= 5].index
        # flag both invoices in a close pair, not just the second
        for idx in close_pairs:
            pos = dates.index.get_loc(idx)
            df.loc[dates.index[pos], "is_potential_duplicate"] = 1
            df.loc[dates.index[pos - 1], "is_potential_duplicate"] = 1

    feature_cols = [
        "invoice_id", "vendor", "category", "amount", "tax_amount",
        "days_to_due", "invoice_month", "invoice_weekday",
        "vendor_avg_amount", "amount_zscore_vendor",
        "missing_field_count", "is_potential_duplicate", "label",
    ]
    return df[feature_cols].reset_index(drop=True)


if __name__ == "__main__":
    from database import InvoiceDB

    raw = pd.read_csv("data/raw_invoices.csv")
    db = InvoiceDB()
    db.load_raw(raw)

    processed = clean_and_engineer(raw)
    db.load_processed(processed)

    print(f"Cleaned {len(processed)} invoices and wrote them to the database.")
    print(processed[["is_potential_duplicate", "missing_field_count"]].sum())
