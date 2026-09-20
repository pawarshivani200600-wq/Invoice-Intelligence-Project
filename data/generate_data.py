"""
generate_data.py
-----------------
Creates a synthetic but realistic invoice dataset.

Why synthetic data: real invoice datasets are usually private company data.
For a portfolio project, we simulate one with the same *shape* and the same
kinds of problems a real dataset has -- duplicate submissions, missing
fields, and abnormal amounts -- so the downstream cleaning + classification
steps are solving a real problem, not a toy one.

Each invoice gets a label (this is what the model will later learn to predict):
    - "Valid"                -> looks normal, safe to process
    - "Duplicate"             -> same vendor + amount + date as another invoice
    - "Amount Anomaly"        -> amount is a statistical outlier for that vendor
    - "Missing Info"          -> required field(s) are blank

Output: data/raw_invoices.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

VENDORS = [
    "Apex Office Supplies", "Bluewave Logistics", "Crestline IT Services",
    "Delta Print Solutions", "Everline Facilities", "Falcon Freight Co",
    "Greenfield Consulting", "Horizon Software Inc", "Ironclad Security",
    "Junction Marketing Group",
]

CATEGORIES = ["Office Supplies", "IT Services", "Logistics", "Consulting",
              "Marketing", "Facilities", "Software License", "Security"]

PAYMENT_TERMS = ["Net 15", "Net 30", "Net 45", "Net 60"]

# Typical amount range per vendor (mean, std) so anomalies are detectable
VENDOR_PROFILE = {v: (np.random.uniform(500, 5000), np.random.uniform(100, 800))
                   for v in VENDORS}

N_BASE_INVOICES = 1400


def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def make_base_invoice(inv_id):
    vendor = random.choice(VENDORS)
    mean, std = VENDOR_PROFILE[vendor]
    amount = round(max(10, np.random.normal(mean, std)), 2)
    invoice_date = random_date(datetime(2025, 1, 1), datetime(2026, 6, 30))
    terms_days = int(random.choice(PAYMENT_TERMS).split(" ")[1])
    due_date = invoice_date + timedelta(days=terms_days)
    tax_amount = round(amount * random.choice([0.05, 0.12, 0.18]), 2)

    return {
        "invoice_id": f"INV-{inv_id:05d}",
        "vendor": vendor,
        "category": random.choice(CATEGORIES),
        "invoice_date": invoice_date.strftime("%Y-%m-%d"),
        "due_date": due_date.strftime("%Y-%m-%d"),
        "payment_terms": random.choice(PAYMENT_TERMS),
        "amount": amount,
        "tax_amount": tax_amount,
        "label": "Valid",
    }


rows = []
for i in range(1, N_BASE_INVOICES + 1):
    rows.append(make_base_invoice(i))

df = pd.DataFrame(rows)

# --- Inject "Amount Anomaly" cases: spike the amount for the vendor's profile ---
anomaly_idx = df.sample(frac=0.07, random_state=RNG_SEED).index
for idx in anomaly_idx:
    vendor = df.loc[idx, "vendor"]
    mean, std = VENDOR_PROFILE[vendor]
    spike_direction = random.choice([1, -1])
    df.loc[idx, "amount"] = round(max(5, mean + spike_direction * random.uniform(5, 9) * std), 2)
    df.loc[idx, "label"] = "Amount Anomaly"

# --- Inject "Missing Info" cases: blank out a required field ---
missing_idx = df.drop(anomaly_idx).sample(frac=0.06, random_state=RNG_SEED + 1).index
for idx in missing_idx:
    field = random.choice(["vendor", "amount", "due_date", "category"])
    df.loc[idx, field] = np.nan
    df.loc[idx, "label"] = "Missing Info"

# --- Inject "Duplicate" cases: clone an existing valid invoice with a new ID ---
dup_source_idx = df[df["label"] == "Valid"].sample(frac=0.08, random_state=RNG_SEED + 2).index
dup_rows = []
next_id = N_BASE_INVOICES + 1
for idx in dup_source_idx:
    clone = df.loc[idx].copy()
    clone["invoice_id"] = f"INV-{next_id:05d}"
    next_id += 1
    # duplicates are usually resubmitted a few days later, same amount/vendor
    d = datetime.strptime(clone["invoice_date"], "%Y-%m-%d") + timedelta(days=random.randint(1, 5))
    clone["invoice_date"] = d.strftime("%Y-%m-%d")
    clone["label"] = "Duplicate"
    dup_rows.append(clone)

df = pd.concat([df, pd.DataFrame(dup_rows)], ignore_index=True)

# Shuffle so labels aren't in blocks
df = df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)

df.to_csv("data/raw_invoices.csv", index=False)
print(f"Generated {len(df)} invoices -> data/raw_invoices.csv")
print(df["label"].value_counts())
