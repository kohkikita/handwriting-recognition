import os
import numpy as np
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.data.dataset_utils import get_emnist_datasets

MODEL_PATH = "models/gradient_model.pkl"

def prepare_data(dataset, max_samples=None):
    """
    Convert PyTorch dataset to numpy arrays for sklearn.
    """
    if max_samples:
        indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
        print(f"Using {len(indices)} samples (subset for faster training)")
    else:
        indices = range(len(dataset))
        print(f"Using full dataset: {len(dataset)} samples")
    
    images = []
    labels = []
    
    for idx in indices:
        img, label = dataset[idx]
        # Flatten: [1, 28, 28] -> [784]
        images.append(img.numpy().flatten())
        labels.append(label)
    
    return np.array(images), np.array(labels)

def main():
    os.makedirs("models", exist_ok=True)
    
    print("Loading EMNIST datasets...")
    train_ds, test_ds = get_emnist_datasets()
    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    
    # Use subset for faster training (GradientBoosting can be slow)
    print("\nPreparing training data...")
    X_train, y_train = prepare_data(train_ds, max_samples=20000)  # Use 20k samples
    print(f"X_train shape: {X_train.shape}")
    
    print("\nPreparing test data...")
    X_test, y_test = prepare_data(test_ds, max_samples=5000)  # Use 5k samples
    print(f"X_test shape: {X_test.shape}")
    
    # Create GradientBoosting model
    # Using smaller n_estimators for faster training
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=1
    )
    
    print("\nTraining GradientBoosting...")
    model.fit(X_train, y_train)
    
    print("\nEvaluating...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=4))
    
    dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()

