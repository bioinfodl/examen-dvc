import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw data into train/test sets.")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw_data/raw.csv"),
        help="Path to raw CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed_data"),
        help="Directory to save split datasets.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    target = "silica_concentrate"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    X = df.drop(columns=[target]) # features
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train.to_csv(args.output_dir / "X_train.csv", index=False)
    X_test.to_csv(args.output_dir / "X_test.csv", index=False)
    y_train.to_csv(args.output_dir / "y_train.csv", index=False, header=True)
    y_test.to_csv(args.output_dir / "y_test.csv", index=False, header=True)


if __name__ == "__main__":
    main()
