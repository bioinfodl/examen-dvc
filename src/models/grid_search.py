import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GridSearch to find best params.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed_data"),
        help="Directory containing processed datasets.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/best_params.pkl"),
        help="Path to save best parameters.",
    )
    parser.add_argument("--cv", type=int, default=3) # number of cross-validation folds
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(args.processed_dir / "X_train_scaled.csv")
    y_train = (
        pd.read_csv(args.processed_dir / "y_train.csv")
        .squeeze("columns")
        .to_numpy()
    )

    model = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=args.cv,
        scoring="neg_mean_squared_error", # use negative MSE for regression
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    joblib.dump(grid_search.best_params_, args.output_path)


if __name__ == "__main__":
    main()
