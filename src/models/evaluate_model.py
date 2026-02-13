import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed_data"),
        help="Directory containing processed datasets.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/model.joblib"),
        help="Path to trained model.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("data/processed_data/predictions.csv"),
        help="Path to save predictions.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("metrics/scores.json"),
        help="Path to save metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(args.processed_dir / "X_test_scaled.csv")
    y_test = pd.read_csv(args.processed_dir / "y_test.csv").squeeze("columns")

    model = joblib.load(args.model_path)
    y_pred = model.predict(X_test)

    predictions = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    predictions.to_csv(args.predictions_path, index=False)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {"mse": mse, "rmse": rmse, "r2": r2, "mae": mae}
    with args.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
