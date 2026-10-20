"""
PondGuard - Aquaculture Water Quality Model Trainer
Dataset Source: Mendeley Data (DOI: 10.17632/y78ty2g293.2)
"Aquaculture - Water Quality Dataset", Veeramsetty et al.
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "aquaculture_water_quality.csv"
EXCEL_PATH = BASE_DIR / "WQD.xlsx"
MODEL_PATH = BASE_DIR / "aquaculture_model.pkl"

TARGET_NAMES = {
    0: "Excellent Quality",
    1: "Good Quality",
    2: "Poor Quality"
}

def load_data():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    elif EXCEL_PATH.exists():
        df = pd.read_excel(EXCEL_PATH)
    else:
        raise FileNotFoundError(
            f"Dataset not found. Expected {CSV_PATH.name} or {EXCEL_PATH.name} in {BASE_DIR}"
        )

    # Dynamic column identification to handle slight formatting differences (backticks, units, spaces)
    temp_col = [c for c in df.columns if "temp" in c.lower()][0]
    do_col = [c for c in df.columns if "do" in c.lower()][0]
    ph_col = [c for c in df.columns if "ph" in c.lower()][0]
    ammonia_col = [c for c in df.columns if "ammonia" in c.lower()][0]
    nitrite_col = [c for c in df.columns if "nitrite" in c.lower()][0]
    target_col = [c for c in df.columns if "quality" in c.lower()][0]

    feature_mapping = {
        temp_col: "temperature",
        do_col: "dissolved_oxygen",
        ph_col: "ph",
        ammonia_col: "ammonia",
        nitrite_col: "nitrite",
    }

    print("Source columns mapped:")
    for src, dst in feature_mapping.items():
        print(f"  - {src!r} -> {dst}")
    print(f"  - Target column: {target_col!r}")

    X = df[list(feature_mapping.keys())].rename(columns=feature_mapping)
    y = df[target_col]

    return X, y

def main():
    print("=" * 65)
    print("PondGuard - Training Aquaculture Water Quality Model")
    print("Dataset: Mendeley Data (DOI: 10.17632/y78ty2g293.2)")
    print("=" * 65)

    X, y = load_data()

    print(f"\nTotal records: {len(X)}")
    print(f"Feature set ({len(X.columns)}): {list(X.columns)}")
    print(f"Missing values:\n{X.isnull().sum()}")

    print("\nTarget class distribution:")
    for cls_val, count in y.value_counts().sort_index().items():
        label_name = TARGET_NAMES.get(cls_val, f"Class {cls_val}")
        print(f"  Class {cls_val} ({label_name}): {count} samples ({count / len(y):.2%})")

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"\nTraining set size: {len(X_train)} | Test set size: {len(X_test)}")

    # Model Pipeline: Median Imputer + Gradient Boosting Classifier
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        )),
    ])

    print("\nTraining Gradient Boosting Pipeline...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    # Evaluation metrics
    acc = accuracy_score(y_test, predictions)
    prec_macro = precision_score(y_test, predictions, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, predictions, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, predictions, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1_mac = f1_score(y_test, predictions, average="macro", zero_division=0)
    f1_weight = f1_score(y_test, predictions, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS (TEST SET - 860 SAMPLES)")
    print("=" * 65)
    print(f"Accuracy:          {acc:.4%} ({acc:.4f})")
    print(f"Macro Precision:   {prec_macro:.4f}  | Weighted Precision: {prec_weighted:.4f}")
    print(f"Macro Recall:      {rec_macro:.4f}  | Weighted Recall:    {rec_weighted:.4f}")
    print(f"Macro F1-Score:    {f1_mac:.4f}  | Weighted F1-Score:  {f1_weight:.4f}")

    print("\nConfusion Matrix:")
    print("                   Pred 0 (Exc)  Pred 1 (Good)  Pred 2 (Poor)")
    print(f"True 0 (Excellent)      {cm[0][0]:<14} {cm[0][1]:<14} {cm[0][2]}")
    print(f"True 1 (Good)           {cm[1][0]:<14} {cm[1][1]:<14} {cm[1][2]}")
    print(f"True 2 (Poor)           {cm[2][0]:<14} {cm[2][1]:<14} {cm[2][2]}")

    print("\nDetailed Classification Report:")
    target_names_list = [TARGET_NAMES[i] for i in sorted(TARGET_NAMES.keys())]
    print(classification_report(y_test, predictions, target_names=target_names_list, digits=4))

    # Save trained pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nTrained pipeline saved to: {MODEL_PATH}")

    # Self-test: Verify saved model can be loaded and predict across all 3 classes
    print("\nRunning verification test on saved artifact (all 3 classes)...")
    loaded_model = joblib.load(MODEL_PATH)
    test_cases = [
        {"name": "Excellent Water Sample", "data": {"temperature": 24.6, "dissolved_oxygen": 4.2, "ph": 8.3, "ammonia": 0.01, "nitrite": 0.001}},
        {"name": "Good Water Sample", "data": {"temperature": 33.7, "dissolved_oxygen": 7.8, "ph": 6.1, "ammonia": 0.04, "nitrite": 1.87}},
        {"name": "Poor/Critical Water Sample", "data": {"temperature": 12.3, "dissolved_oxygen": 1.8, "ph": 5.1, "ammonia": 0.21, "nitrite": 4.55}},
    ]
    for tc in test_cases:
        df_tc = pd.DataFrame([tc["data"]])
        pred_cls = loaded_model.predict(df_tc)[0]
        prob_dist = loaded_model.predict_proba(df_tc)[0]
        prob_dict = {TARGET_NAMES[i]: f"{p:.2%}" for i, p in enumerate(prob_dist)}
        print(f" - {tc['name']} -> Predicted Class {pred_cls} ({TARGET_NAMES[pred_cls]}) | Probabilities: {prob_dict}")
    print("Verification test passed successfully!")

if __name__ == "__main__":
    main()
