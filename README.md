# Invoice Intelligence — ML-Powered Predictive Classification System

An end-to-end system that classifies incoming invoices as **Valid**,
**Duplicate**, **Amount Anomaly**, or **Missing Info**, so a finance/ops
team can review only the invoices that actually need a human look instead
of every single one.

This matches the resume bullet:
> Building a predictive model in Scikit-Learn to classify and validate
> invoice data, including a full data cleaning and feature engineering
> pipeline. Designing a SQL-backed data layer to structure extracted data
> for model training and evaluation. Deploying the model through an
> interactive Streamlit app to surface predictions and insights.

## How it works, step by step

### 1. Data (`data/generate_data.py`)
Real invoice data is private company data, so this project simulates a
realistic dataset: ~1,500 invoices across 10 vendors, with intentionally
injected problems — duplicate resubmissions, blanked-out required fields,
and vendor-abnormal amounts — labeled accordingly. This mirrors what a
real accounts-payable dataset looks like: mostly fine, with a minority of
genuinely problematic invoices.
To use your **own** data instead, just replace `data/raw_invoices.csv`
with a CSV that has the same columns.

### 2. SQL-backed data layer (`database.py`)
A SQLite database (`invoices.db`) stores three tables:
- `raw_invoices` — data as received
- `processed_invoices` — cleaned data + engineered features + label
- `predictions` — every prediction the app makes on new invoices, logged
  with a timestamp

Using a real database (instead of just passing CSVs/DataFrames around)
means the data is queryable with SQL directly — `database.py` includes a
`vendor_summary()` method that does a `GROUP BY`/`AVG`/`SUM` aggregation
in SQL rather than pandas, which is the kind of thing you'd actually be
asked to do in a data role.

### 3. Cleaning + feature engineering (`pipeline.py`)
Raw invoices are turned into model-ready features:
- **`missing_field_count`** — how many required fields are blank
- **`amount_zscore_vendor`** — how many standard deviations this
  invoice's amount is from *that specific vendor's* normal amount
  (a $50,000 invoice might be normal for one vendor and wildly
  abnormal for another — comparing to the global average would miss that)
- **`is_potential_duplicate`** — same vendor + amount, submitted within
  5 days of another invoice
- `days_to_due`, `invoice_month`, `invoice_weekday` — timing signals

### 4. Model training (`train_model.py`)
A `RandomForestClassifier` (scikit-learn `Pipeline` with one-hot encoding
for category) trained with `class_weight="balanced"` — because in real
data, the vast majority of invoices are valid, and without balancing the
model could get high accuracy by just never flagging anything.
Evaluated with per-class precision/recall/F1 (not just accuracy, which is
misleading on imbalanced data) and a confusion matrix. Current run:
**~96% accuracy, 0.93 macro F1** on held-out test data, with
`missing_field_count`, `is_potential_duplicate`, and `amount_zscore_vendor`
as the three most important features — i.e., the model is keying off
exactly the signals it should be.

### 5. Streamlit app (`app/streamlit_app.py`)
Three tabs:
- **Classify New Invoices** — upload a CSV (or try a sample), see each
  invoice's predicted label + confidence, flagged rows highlighted
- **Vendor Insights** — per-vendor invoice volume, average amount, and
  flagged-invoice count, computed in SQL
- **Training Data Overview** — label distribution and a peek at the
  feature-engineered training table

## Running it yourself

```bash
pip install -r requirements.txt

# 1. generate the synthetic dataset
python data/generate_data.py

# 2. clean it, engineer features, load into SQLite
python pipeline.py

# 3. train and evaluate the model
python train_model.py

# 4. launch the app
streamlit run app/streamlit_app.py
```

## Project structure
```
invoice-intelligence/
├── data/
│   ├── generate_data.py     # synthetic dataset generator
│   └── raw_invoices.csv     # generated dataset
├── app/
│   └── streamlit_app.py     # the deployed front end
├── database.py               # SQLite data layer
├── pipeline.py                # cleaning + feature engineering
├── train_model.py             # training + evaluation
├── models/
│   └── invoice_classifier.joblib
├── invoices.db                 # SQLite database file
├── requirements.txt
└── README.md
```

## Talking about this in an interview
If asked to walk through it: start with *why* — flagging every invoice
for manual review doesn't scale, so the goal is a model that narrows
attention to the ~13% that actually need it. Then walk the pipeline in
order (raw → SQL → cleaned features → model → app), and be ready to
explain **why** `amount_zscore_vendor` is computed per-vendor rather than
globally, and why `class_weight="balanced"` matters given the label
imbalance — those two choices are the ones most likely to get a follow-up
question.
