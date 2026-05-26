"""
Train gesture classifier from collected samples.

Run: python gesture_trainer.py
Output: gesture_model.pkl + gesture_classes.txt

Requirements: scikit-learn, joblib, numpy (all included in pip install scikit-learn)
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tf_gesture import GESTURE_CLASSES, normalize_lm

DATA_FILE    = os.path.join(os.path.dirname(__file__), "gesture_data", "gestures.csv")
MODEL_FILE   = os.path.join(os.path.dirname(__file__), "gesture_model.pkl")
CLASSES_FILE = os.path.join(os.path.dirname(__file__), "gesture_classes.txt")


def load_data():
    label_to_idx = {cls: i for i, cls in enumerate(GESTURE_CLASSES)}
    X, y = [], []
    skipped = 0
    with open(DATA_FILE, "r", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] not in label_to_idx:
                continue
            try:
                features = [float(v) for v in row[1:]]
            except ValueError:
                skipped += 1
                continue
            if len(features) != 63:
                skipped += 1
                continue
            X.append(features)
            y.append(label_to_idx[row[0]])
    if skipped:
        print(f"  {skipped} linhas ignoradas (formato inválido)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train():
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report

    print(f"\nCarregando {DATA_FILE}...")
    X, y = load_data()
    print(f"  {len(X)} amostras totais")

    counts = {cls: int((y == i).sum()) for i, cls in enumerate(GESTURE_CLASSES)}
    min_samples = min(counts.values())
    for cls, n in counts.items():
        bar = "█" * (n // 10)
        warn = " ⚠ poucos" if n < 100 else ""
        print(f"  {cls:15s} {n:4d}  {bar}{warn}")

    if min_samples < 30:
        print(f"\nAVISO: classe com apenas {min_samples} amostras — colete mais antes de treinar.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"\nTreino: {len(X_train)}  Validação: {len(X_val)}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,          # L2 regularization
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
            verbose=True,
        )),
    ])

    print("\nTreinando...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = (y_pred == y_val).mean()
    print(f"\nAcurácia validação: {acc * 100:.1f}%")
    print("\n" + classification_report(
        y_val, y_pred,
        target_names=GESTURE_CLASSES,
        zero_division=0,
    ))

    if acc < 0.85:
        print("AVISO: acurácia < 85% — colete mais amostras variadas e retreine.")

    joblib.dump(model, MODEL_FILE)
    print(f"Modelo salvo: {MODEL_FILE}  ({os.path.getsize(MODEL_FILE) / 1024:.1f} KB)")

    with open(CLASSES_FILE, "w") as f:
        f.write("\n".join(GESTURE_CLASSES))
    print(f"Classes salvas: {CLASSES_FILE}")
    print("\nPara usar: reinicie python main.py — modelo carregará automaticamente.")


if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"Dados não encontrados: {DATA_FILE}")
        print("Execute primeiro: python gesture_collector.py")
        sys.exit(1)
    train()
