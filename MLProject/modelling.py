import os
import argparse
import pandas as pd
import numpy as np 
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import json

# --- Argument parser for MLflow Project ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="dataset_preprocessing")
args = parser.parse_args()

# --- Load dataset ---
X_train = pd.read_csv(f"{args.data_path}/train_processed.csv")
X_test = pd.read_csv(f"{args.data_path}/test_processed.csv")
y_train = pd.read_csv(f"{args.data_path}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{args.data_path}/y_test.csv").values.ravel()

# --- Candidate models + hyperparameter grids ---
models = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1.0, 10.0]}
    ),
    "RandomForest": (
        RandomForestClassifier(),
        {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
    )
}

# --- Run experiments ---
for model_name, (estimator, param_grid) in models.items():
    with mlflow.start_run(nested=True):
        grid = GridSearchCV(estimator, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # --- Metrics ---
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred)

        # --- Log params & metrics ---
        mlflow.log_param("model", model_name)
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # --- Confusion matrix ---
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        tmpdir = tempfile.mkdtemp()
        cm_path = os.path.join(tmpdir, f"confusion_matrix_{model_name}.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # --- Classification report ---
        report_path = os.path.join(tmpdir, f"classification_report_{model_name}.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path="reports")

        # --- Save best model ---
        mlflow.sklearn.log_model(best_model, artifact_path=f"models/{model_name}")

        # --- Save grid search results ---
        cv_results_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in grid.cv_results_.items()
        }
        results_path = os.path.join(tmpdir, f"grid_results_{model_name}.json")
        with open(results_path, "w") as f:
            json.dump(cv_results_serializable, f, indent=4)
        mlflow.log_artifact(results_path, artifact_path="grid_search")

        print(f"✅ {model_name} Best Accuracy: {acc:.4f}, Best Params: {grid.best_params_}")