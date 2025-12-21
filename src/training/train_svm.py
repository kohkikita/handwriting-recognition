import os
import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report
from src.data.dataset_utils import get_emnist_datasets
from src.models.svm_model import build_svm

MODEL_PATH = "models/svm_emnist.joblib"

def main():
    os.makedirs("models", exist_ok=True)

    train_ds, test_ds = get_emnist_datasets()

    # EMNIST datasets store raw data as uint8 images in train_ds.data (H,W)
    X_train = train_ds.data.numpy().astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy()

    X_test = test_ds.data.numpy().astype(np.float32) / 255.0
    y_test = test_ds.targets.numpy()

    # Flatten for SVM
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    svm = build_svm()
    print("Training SVM...")
    svm.fit(X_train, y_train)

    print("Evaluating...")
    preds = svm.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Test Accuracy:", acc)
    print(classification_report(y_test, preds, digits=4))

    dump(svm, MODEL_PATH)
    print("Saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
