"""
CatchAI — MVP Baseline
- Loads your fishing log CSV
- Engineers simple time features
- Trains a quick logistic regression
- Prints a “best hour” table
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib

DATA_PATH = Path("catchai_dataset_template.csv")
MODEL_PATH = Path("catchai_model.pkl")

CATEGORICAL = [
    "location_name","water_body","species","method","lure","lure_color",
    "retrieve_style","pressure_trend","structure","cover","bottom_type","moon_phase","tide_stage"
]
NUMERIC = [
    "air_temp_f","water_temp_f","wind_mph","wind_direction_deg",
    "barometer_inHg","cloud_cover_pct","precip_in_last24h",
    "water_clarity_ft","depth_ft","moon_illumination_pct","latitude","longitude"
]
TARGET = "caught_fish"

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    df = pd.read_csv(path)

    # Cast numeric cols
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Target to int 0/1
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    else:
        df[TARGET] = 0

    # Fill categoricals
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
        else:
            df[col] = "unknown"

    # Time features
    if "datetime_iso" in df.columns:
        dt = pd.to_datetime(df["datetime_iso"], errors="coerce")
        df["hour"] = dt.dt.hour.fillna(6).astype(int)
        df["month"] = dt.dt.month.fillna(6).astype(int)
        df["weekday"] = dt.dt.weekday.fillna(5).astype(int)
    else:
        df["hour"] = 6
        df["month"] = 6
        df["weekday"] = 5

    return df

def train_model(df: pd.DataFrame):
    features = CATEGORICAL + NUMERIC + ["hour","month","weekday"]
    X = df[features]
    y = df[TARGET]

    pre = ColumnTransformer([
    ("cat",
     OneHotEncoder(handle_unknown="ignore"),
     CATEGORICAL + ["hour","month","weekday"]),
    ("num",
     Pipeline([
         ("impute", SimpleImputer(strategy="median")),
         ("scale", StandardScaler())
     ]),
     NUMERIC)
])
    clf = LogisticRegression(max_iter=200)
    pipe = Pipeline([("preprocess", pre), ("clf", clf)])

    # Tiny dataset safeguard
    class_counts = y.value_counts(dropna=False)
    too_small = (len(df) < 6) or (class_counts.min() < 2)

    if too_small:
        print("Tiny dataset mode: training on all rows (skipping train/test split & metrics).")
        pipe.fit(X, y)
        joblib.dump(pipe, MODEL_PATH)
        print(f"Saved model -> {MODEL_PATH.resolve()}")
        return pipe

    # Normal path with stratified split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe.fit(Xtr, ytr)
    yprob = pipe.predict_proba(Xte)[:, 1]
    try:
        auc = roc_auc_score(yte, yprob)
        print(f"AUC: {auc:.3f}")
    except ValueError:
        print("AUC: N/A (not enough class variety)")
    print(classification_report(yte, pipe.predict(Xte)))
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH.resolve()}")
    return pipe

def recommend_times(pipe, df_row: pd.Series, start=5, end=21, topk=5):
    """Vary the hour across same conditions and rank."""
    features = CATEGORICAL + NUMERIC + ["hour","month","weekday"]
    base = df_row.to_dict()
    test_grid = [{**base, "hour": h} for h in range(start, end + 1)]
    df_test = pd.DataFrame(test_grid)
    preds = pipe.predict_proba(df_test[features])[:, 1]
    df_test["catch_prob"] = preds
    return df_test.sort_values("catch_prob", ascending=False).head(topk)[["hour","catch_prob"]]

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    if len(df) < 20:
        print("Tip: add ~50–100 rows over time for better signals (include skunks!).")
    model = train_model(df)
    if model is not None:
        best = recommend_times(model, df.iloc[0])
        print("\nTop hours for a similar setup:")
        print(best.to_string(index=False))
