import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize train/test features.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed_data"),
        help="Directory containing split datasets.",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=Path("models/scaler.joblib"),
        help="Path to save the fitted scaler.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.scaler_path.parent.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(args.processed_dir / "X_train.csv")
    X_test = pd.read_csv(args.processed_dir / "X_test.csv")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # fit on train only, not on test
    X_test_scaled = scaler.transform(X_test) 

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    X_train_scaled_df.to_csv(args.processed_dir / "X_train_scaled.csv", index=False)
    X_test_scaled_df.to_csv(args.processed_dir / "X_test_scaled.csv", index=False)

    joblib.dump(scaler, args.scaler_path)


if __name__ == "__main__":
    main()
