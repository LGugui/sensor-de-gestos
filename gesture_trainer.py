"""
Train gesture classifier from collected samples.

Run: python gesture_trainer.py
Output: gesture_model.tflite + gesture_classes.txt

Requirements:
  pip install tensorflow
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tf_gesture import GESTURE_CLASSES, normalize_lm

DATA_FILE = os.path.join(os.path.dirname(__file__), "gesture_data", "gestures.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "gesture_model.tflite")
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
        print(f"  {skipped} linhas ignoradas (formato invalido)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras

    print(f"TensorFlow {tf.__version__}")
    print(f"\nCarregando {DATA_FILE}...")
    X, y = load_data()
    print(f"  {len(X)} amostras totais")

    counts = {cls: int((y == i).sum()) for i, cls in enumerate(GESTURE_CLASSES)}
    min_samples = min(counts.values())
    for cls, n in counts.items():
        bar = "#" * (n // 10)
        warn = " ⚠ poucos" if n < 100 else ""
        print(f"  {cls:15s} {n:4d}  {bar}{warn}")

    if min_samples < 30:
        print(f"\nAVISO: classe com apenas {min_samples} amostras — treine mais antes.")

    # One-hot
    y_onehot = keras.utils.to_categorical(y, len(GESTURE_CLASSES))

    # Shuffle + split 80/20
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.8)
    X_tr, X_val = X[idx[:split]], X[idx[split:]]
    y_tr, y_val = y_onehot[idx[:split]], y_onehot[idx[split:]]

    print(f"\nTreino: {len(X_tr)}  Validacao: {len(X_val)}")

    model = keras.Sequential([
        keras.layers.Input(shape=(63,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(GESTURE_CLASSES), activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\nTreinando...")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=15, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                patience=7, factor=0.5, verbose=1),
        ],
    )

    _, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nAcuracia validacao: {acc * 100:.1f}%")

    if acc < 0.85:
        print("AVISO: acuracia < 85% — colete mais amostras e retreine.")

    # Export TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    with open(MODEL_FILE, "wb") as f:
        f.write(tflite_bytes)
    print(f"\nModelo salvo: {MODEL_FILE}  ({len(tflite_bytes) / 1024:.1f} KB)")

    with open(CLASSES_FILE, "w") as f:
        f.write("\n".join(GESTURE_CLASSES))
    print(f"Classes salvas: {CLASSES_FILE}")

    print("\nPara usar: reinicie o sensor — ele carregara o modelo automaticamente.")


if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"Dados nao encontrados: {DATA_FILE}")
        print("Execute primeiro: python gesture_collector.py")
        sys.exit(1)
    train()
