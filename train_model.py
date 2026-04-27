from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_rf_tuned.joblib"

CATEGORICAL_COLUMNS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC_COLUMNS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    x = df[FEATURE_COLUMNS]
    y = df["HeartDisease"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [4, 8, None],
            "classifier__min_samples_leaf": [1, 3, 5],
        },
        cv=5,
        scoring="f1",
        n_jobs=1,
    )
    search.fit(x_train, y_train)

    model = search.best_estimator_
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Best parameters:", search.best_params_)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("F1:", round(f1_score(y_test, predictions), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    print(classification_report(y_test, predictions))
    print("Saved model:", MODEL_PATH)


if __name__ == "__main__":
    main()
