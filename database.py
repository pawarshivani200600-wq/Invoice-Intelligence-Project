"""
database.py
-----------
SQL-backed data layer for the invoice pipeline.

Uses SQLite (file-based SQL database, no server needed -- easy to demo,
same SQL concepts as Postgres/MySQL). This module is the single place
that talks to the database: raw invoices go in, cleaned/feature-engineered
invoices go in, and the model reads from here for training and inference.

Tables:
    raw_invoices       -- invoices exactly as received (may have nulls, dupes)
    processed_invoices -- cleaned invoices + engineered features + label
    predictions         -- model predictions logged for new/unseen invoices
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / "invoices.db"


class InvoiceDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_invoices (
                    invoice_id TEXT PRIMARY KEY,
                    vendor TEXT,
                    category TEXT,
                    invoice_date TEXT,
                    due_date TEXT,
                    payment_terms TEXT,
                    amount REAL,
                    tax_amount REAL,
                    label TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_invoices (
                    invoice_id TEXT PRIMARY KEY,
                    vendor TEXT,
                    category TEXT,
                    amount REAL,
                    tax_amount REAL,
                    days_to_due INTEGER,
                    invoice_month INTEGER,
                    invoice_weekday INTEGER,
                    vendor_avg_amount REAL,
                    amount_zscore_vendor REAL,
                    missing_field_count INTEGER,
                    is_potential_duplicate INTEGER,
                    label TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    invoice_id TEXT PRIMARY KEY,
                    predicted_label TEXT,
                    confidence REAL,
                    predicted_at TEXT
                )
            """)

    def load_raw(self, df: pd.DataFrame):
        with self._connect() as conn:
            df.to_sql("raw_invoices", conn, if_exists="replace", index=False)

    def load_processed(self, df: pd.DataFrame):
        with self._connect() as conn:
            df.to_sql("processed_invoices", conn, if_exists="replace", index=False)

    def log_predictions(self, df: pd.DataFrame):
        with self._connect() as conn:
            df.to_sql("predictions", conn, if_exists="append", index=False)

    def get_raw(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql("SELECT * FROM raw_invoices", conn)

    def get_processed(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql("SELECT * FROM processed_invoices", conn)

    def get_predictions(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql("SELECT * FROM predictions", conn)

    def vendor_summary(self) -> pd.DataFrame:
        """Example of doing aggregation in SQL rather than pandas --
        useful to show in an interview: GROUP BY, AVG, COUNT."""
        with self._connect() as conn:
            return pd.read_sql("""
                SELECT vendor,
                       COUNT(*) AS invoice_count,
                       ROUND(AVG(amount), 2) AS avg_amount,
                       SUM(CASE WHEN label != 'Valid' THEN 1 ELSE 0 END) AS flagged_count
                FROM processed_invoices
                GROUP BY vendor
                ORDER BY flagged_count DESC
            """, conn)
