"""Programmatically generate all dirty / clean CSV pairs.

Usage::

    python -m scripts.generate_datasets      # from project root
    python scripts/generate_datasets.py       # alternative

The script is **deterministic** (``seed=42``) so that every run produces
identical files.  It writes six CSVs into ``data/{easy,medium,hard}/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ── constants ────────────────────────────────────────────────────────
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

fake = Faker()
Faker.seed(SEED)
np.random.seed(SEED)


# =====================================================================
#  EASY — 10-row employee CSV with 5 missing values
# =====================================================================

def _generate_easy() -> None:
    """Generate the easy task CSVs: employee table with missing values."""
    out_dir = DATA_DIR / "easy"
    out_dir.mkdir(parents=True, exist_ok=True)

    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance"]
    rows: list[dict] = []
    for i in range(10):
        rows.append(
            {
                "employee_id": i + 1,
                "name": fake.name(),
                "age": np.random.randint(22, 60),
                "department": np.random.choice(departments),
                "salary": round(np.random.uniform(40_000, 120_000), 2),
                "email": fake.email(),
            }
        )

    clean_df = pd.DataFrame(rows)
    clean_df.to_csv(out_dir / "clean.csv", index=False)

    # Inject 5 missing values at deterministic positions
    dirty_df = clean_df.copy()
    missing_positions = [
        (1, "salary"),
        (3, "department"),
        (5, "age"),
        (7, "salary"),
        (9, "department"),
    ]
    for row_idx, col in missing_positions:
        dirty_df.at[row_idx, col] = np.nan

    dirty_df.to_csv(out_dir / "dirty.csv", index=False)
    print(f"  ✓ easy  — {out_dir}")


# =====================================================================
#  MEDIUM — 30-row sales CSV with type errors + duplicates
# =====================================================================

def _generate_medium() -> None:
    """Generate the medium task CSVs: sales with types & duplicates."""
    out_dir = DATA_DIR / "medium"
    out_dir.mkdir(parents=True, exist_ok=True)

    products = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Doohickey"]
    regions = ["North", "South", "East", "West"]
    rows: list[dict] = []
    for i in range(25):  # 25 unique rows — we will add 5 duplicates
        rows.append(
            {
                "sale_id": i + 1,
                "product": np.random.choice(products),
                "quantity": np.random.randint(1, 50),
                "price": round(np.random.uniform(10, 1500), 2),
                "date": fake.date_between(start_date="-1y", end_date="today").isoformat(),
                "region": np.random.choice(regions),
            }
        )

    clean_df = pd.DataFrame(rows)
    clean_df.to_csv(out_dir / "clean.csv", index=False)

    # ----- Build dirty copy -----
    dirty_df = clean_df.copy()

    # 1) Format price as currency string "$1,200.00"
    dirty_df["price"] = dirty_df["price"].apply(lambda v: f"${v:,.2f}")

    # 2) Mix date formats
    date_formats = ["%m/%d/%Y", "%d-%b-%Y", "%Y.%m.%d", "%B %d, %Y"]
    for idx in dirty_df.index:
        dt = pd.to_datetime(clean_df.at[idx, "date"])
        fmt = date_formats[idx % len(date_formats)]
        dirty_df.at[idx, "date"] = dt.strftime(fmt)

    # 3) Inject negative quantities at 5 positions
    neg_rows = [2, 8, 14, 19, 23]
    for r in neg_rows:
        dirty_df.at[r, "quantity"] = -abs(dirty_df.at[r, "quantity"])

    # 4) Add 5 duplicate rows (copies of rows 0, 5, 10, 15, 20)
    dup_rows = dirty_df.iloc[[0, 5, 10, 15, 20]].copy()
    dirty_df = pd.concat([dirty_df, dup_rows], ignore_index=True)

    dirty_df.to_csv(out_dir / "dirty.csv", index=False)
    print(f"  ✓ medium — {out_dir}")


# =====================================================================
#  HARD — 100-row customer transaction CSV, multiple error categories
# =====================================================================

def _generate_hard() -> None:
    """Generate the hard task CSVs: full pipeline with mixed errors."""
    out_dir = DATA_DIR / "hard"
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = ["Electronics", "Clothing", "Food", "Books", "Toys"]
    countries_clean = ["USA", "Canada", "UK", "Germany", "France"]
    payment_methods = ["Credit Card", "Debit Card", "PayPal", "Cash", "Wire Transfer"]

    rows: list[dict] = []
    for i in range(100):
        rows.append(
            {
                "transaction_id": f"TXN-{i+1:04d}",
                "customer_name": fake.name(),
                "email": fake.email(),
                "country": np.random.choice(countries_clean),
                "category": np.random.choice(categories),
                "amount": round(np.random.uniform(5, 5000), 2),
                "quantity": np.random.randint(1, 20),
                "date": fake.date_between(start_date="-2y", end_date="today").isoformat(),
                "payment_method": np.random.choice(payment_methods),
                "rating": np.random.randint(1, 6),  # 1-5
            }
        )

    clean_df = pd.DataFrame(rows)
    clean_df.to_csv(out_dir / "clean.csv", index=False)

    dirty_df = clean_df.copy()

    # Convert columns that will hold mixed types to object dtype first
    dirty_df["amount"] = dirty_df["amount"].astype(object)
    dirty_df["quantity"] = dirty_df["quantity"].astype(object)
    dirty_df["rating"] = dirty_df["rating"].astype(object)

    # 1) Missing values — ~10 % of cells across selected columns
    missing_cols = ["customer_name", "email", "amount", "quantity", "rating"]
    rng = np.random.default_rng(SEED)
    for col in missing_cols:
        mask = rng.random(len(dirty_df)) < 0.10
        dirty_df.loc[mask, col] = np.nan

    # 2) Type mismatches — amount as currency strings for 15 rows
    type_rows = rng.choice(len(dirty_df), size=15, replace=False)
    for r in type_rows:
        val = dirty_df.at[r, "amount"]
        if pd.notna(val):
            dirty_df.at[r, "amount"] = f"${float(val):,.2f}"

    # 3) Duplicate rows — 8 duplicates
    dup_indices = rng.choice(len(clean_df), size=8, replace=False)
    dup_rows = dirty_df.iloc[dup_indices].copy()
    dirty_df = pd.concat([dirty_df, dup_rows], ignore_index=True)

    # 4) Outliers (3-sigma) — inject 5 extreme amounts
    mean_amt = clean_df["amount"].mean()
    std_amt = clean_df["amount"].std()
    outlier_rows = [3, 22, 45, 67, 88]
    for r in outlier_rows:
        dirty_df.at[r, "amount"] = round(mean_amt + 4 * std_amt, 2)

    # 5) Inconsistent categorical values for 'country'
    country_variants = {
        "USA": ["US", "U.S.A", "United States", "usa"],
        "UK": ["U.K.", "United Kingdom", "uk"],
        "Canada": ["CA", "canada"],
        "Germany": ["DE", "germany"],
        "France": ["FR", "france"],
    }
    for idx in dirty_df.index:
        original = clean_df.at[idx, "country"] if idx < len(clean_df) else dirty_df.at[idx, "country"]
        if original in country_variants and rng.random() < 0.35:
            dirty_df.at[idx, "country"] = rng.choice(country_variants[original])

    dirty_df.to_csv(out_dir / "dirty.csv", index=False)
    print(f"  ✓ hard  — {out_dir}")


# =====================================================================
#  ENTRYPOINT
# =====================================================================

def generate_all() -> None:
    """Generate every dataset pair (dirty + clean) for all difficulties."""
    print("Generating datasets …")
    _generate_easy()
    _generate_medium()
    _generate_hard()
    print("Done ✓")


if __name__ == "__main__":
    generate_all()
