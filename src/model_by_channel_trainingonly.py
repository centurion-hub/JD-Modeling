import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "demo_training_dataset.csv"

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess date, numeric and categorical features."""

    # --- Convert date fields to year and month ---
    for col in ["activate_date", "deactivate_date", "first_order_month"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"]  = df[col].dt.year.fillna(-1).astype(int)
            df[f"{col}_month"] = df[col].dt.month.fillna(-1).astype(int)
            df.drop(columns=[col], inplace=True)

    # --- Convert numeric fields ---
    for col in ["attribute1", "attribute2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

    # --- Convert all object-type columns to categorical ---
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype("category")

    return df

def main():
    # === 0. Load dataset (must contain channel) ===
    df = pd.read_csv(DATA_FILE)
    df = prepare_data(df)

    if "channel" not in df.columns:
        raise ValueError("channel column not found")

    # === 1. Detect unique channels ===
    channels = sorted(df["channel"].dropna().unique())
    results = {}

    # === 2. Train a model for each channel ===
    for ch in channels:
        subset = df[df["channel"] == ch]
        if subset["purchase"].nunique() < 2:
            print(f"Skip channel {ch}: not enough samples")
            continue

        drop_cols = ["user_ID", "sku_ID", "purchase", "request_time", "channel", "hour", "weekday"]
        X = subset.drop(columns=[col for col in drop_cols if col in subset.columns])
        y = subset["purchase"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Detect categorical feature indices
        cat_feats = [i for i, c in enumerate(X_train.columns)
                     if str(X_train[c].dtype) == "category"]

        # Train LightGBM classifier
        model = LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(X_train, y_train, categorical_feature=cat_feats)

        # Evaluate AUC
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

        # Feature importance
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        top_feats = importances.sort_values(ascending=False).head(10)

        results[ch] = {"auc": auc, "top_features": top_feats}

        print(f"\nChannel: {ch} | AUC: {auc:.4f}")
        print(top_feats)

    # === 3. Example plot: top features of the first channel ===
    if results:
        sample_channel = list(results.keys())[0]
        plt.figure(figsize=(8, 6))
        results[sample_channel]["top_features"].sort_values().plot(kind="barh")
        plt.title(f"Top-10 Feature Importances (Channel: {sample_channel})")
        plt.tight_layout()

        out = ROOT / "results" / f"channel_{sample_channel}_top_features.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        print("Saved feature importance plot:", out)

if __name__ == "__main__":
    main()
