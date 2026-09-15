from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from model_utils import load_dataset, make_features


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "dataset" / "cybersecurity.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "cybersecurity_model.joblib"
RANDOM_STATE = 42


def build_pipeline(X):
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=400,
                    class_weight="balanced",
                    max_features=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():
    df = load_dataset(DATA_PATH)
    X = make_features(df)
    y = df["label"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    model = build_pipeline(X)
    model.fit(X_train, y_train)

    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_name": "ExtraTreesClassifier",
            "feature_version": 1,
            "trained_rows": len(X_train),
        },
        ARTIFACT_PATH,
        compress=3,
    )
    print(f"Saved model artifact to {ARTIFACT_PATH}")
    print(f"Training rows: {len(X_train):,}")


if __name__ == "__main__":
    main()
