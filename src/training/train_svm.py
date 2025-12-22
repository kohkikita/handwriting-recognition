import os
import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report
from src.data.dataset_utils import get_emnist_datasets
from src.models.svm_mode import build_svm

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
    
    # Use subset for faster training (SVM can be slow on full dataset)
    print(f"Full dataset: Train={len(X_train)}, Test={len(X_test)}")
    print("Using subset for faster training...")
    train_size = min(10000, len(X_train))  # Use 10k samples
    test_size = min(2000, len(X_test))     # Use 2k samples
    
    indices_train = np.random.choice(len(X_train), train_size, replace=False)
    indices_test = np.random.choice(len(X_test), test_size, replace=False)
    
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

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
