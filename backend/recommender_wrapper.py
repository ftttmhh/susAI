import pandas as pd
import numpy as np
from recommender import recommend  # assumes recommender.recommend exists

CSV_PATH = "model_metadata.csv"

# Map CSV columns to expected names for recommender.py
COLUMN_MAP = {
    "Model": "model",
    "model": "model",
    "Provider": "provider",
    "provider": "provider",
    "Task": "task",
    "Task Name": "task",
    "task": "task",
    "Parameters": "parameters",
    "Parameters #": "parameters",
    "gpu_energy": "gpu_energy",
    "GPU Energy": "gpu_energy",
    "Latency": "latency_ms",
    "latency_ms": "latency_ms",
    "Latency (ms)": "latency_ms",
    "Accuracy": "accuracy",
    "accuracy": "accuracy",
    "kg_co2_per_kwh": "co2_kg_per_1k",
    "CO2 kg per 1k": "co2_kg_per_1k",
    "kg_co2_per_1k": "co2_kg_per_1k",
    "Score": "score",
}

NUMERIC_COLS = ["gpu_energy", "latency_ms", "accuracy", "co2_kg_per_1k"]


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    """Attempt to clean numeric series with common formatting issues (commas, whitespace).
    Returns a float Series with NaN where conversion fails.
    """
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        # remove commas and non-numeric characters (but keep dots and minus)
        cleaned = s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        return pd.to_numeric(cleaned.replace(["", "nan", "NaN"], np.nan), errors="coerce")
    else:
        return pd.to_numeric(s, errors="coerce")


def load_metadata(csv_path: str = CSV_PATH):
    # Try utf-8-sig, fallback to latin1 if needed
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")

    # Normalize column names to remove leading/trailing spaces
    df.columns = [str(c).strip() for c in df.columns]

    # Rename columns to match expected names
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_MAP:
            rename_map[col] = COLUMN_MAP[col]
    df = df.rename(columns=rename_map)

    # Ensure all expected columns exist
    for col in set(COLUMN_MAP.values()):
        if col not in df.columns:
            print(f"[WARN] Column missing in CSV: {col}")
            df[col] = np.nan

    # Clean numeric columns with helper
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = _clean_numeric_series(df[col])

    # Keep model and task as strings
    if "model" in df.columns:
        df["model"] = df["model"].astype(str)
    if "task" in df.columns:
        df["task"] = df["task"].astype(str)

    return df

def recommend_models(task, min_acc, max_lat, carbon_budget, priority, topk=3, baseline_model=None):
    df = load_metadata()
    out, ctx = recommend(
        df,
        task=task,
        min_acc=min_acc,
        max_lat=max_lat,
        carbon_budget=carbon_budget,
        priority=priority,
        w_acc=None, w_energy=None, w_carbon=None, w_lat=None,
        topk=topk,
        baseline_model=baseline_model
    )
    # 'out' may be DataFrame or list depending on recommend implementation
    if hasattr(out, "to_dict"):
        records = out.to_dict(orient="records")
    elif isinstance(out, list):
        records = out
    else:
        # fallback: convert to empty list
        try:
            records = list(out)
        except Exception:
            records = []
    return records, ctx
