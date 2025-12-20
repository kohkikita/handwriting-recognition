import numpy as np
from src.data.dataset_utils import get_emnist_datasets, get_label_list
from src.models.knearest_model import KNearestModel
import os

def prepare_data(dataset, max_samples=None):
    """
    Convert PyTorch dataset to numpy arrays for sklearn.
    
    Args:
        dataset: PyTorch dataset (already has transforms applied)
        max_samples: Optional limit for faster testing (use None for full dataset)
    
    Returns:
        X: numpy array of flattened images, shape (n_samples, 784)
        y: numpy array of labels, shape (n_samples,)
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
        img, label = dataset[idx]  # Already transformed by dataset_utils
        # Flatten: [1, 28, 28] -> [784]
        images.append(img.numpy().flatten())
        labels.append(label)
    
    return np.array(images), np.array(labels)

def train_knn(n_neighbors=5, train_subset=None, test_subset=None):
    """
    Train and evaluate K-Nearest Neighbors model.
    
    Args:
        n_neighbors: Number of neighbors for KNN
        train_subset: Max training samples (None = use all)
        test_subset: Max test samples (None = use all)
    """
    print("="*50)
    print("K-Nearest Neighbors Training")
    print("="*50)
    
    # Step 1: Load datasets (dataset_utils handles download & transforms)
    print("\n[1/6] Loading EMNIST datasets...")
    train_ds, test_ds = get_emnist_datasets()
    print(f"   Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    
    # Step 2: Prepare training data
    print("\n[2/6] Preparing training data...")
    X_train, y_train = prepare_data(train_ds, max_samples=train_subset)
    print(f"   X_train shape: {X_train.shape}")
    
    # Step 3: Prepare test data
    print("\n[3/6] Preparing test data...")
    X_test, y_test = prepare_data(test_ds, max_samples=test_subset)
    print(f"   X_test shape: {X_test.shape}")
    
    # Step 4: Create and train model
    print(f"\n[4/6] Training KNN (n_neighbors={n_neighbors})...")
    model = KNearestModel(n_neighbors=n_neighbors, weights='distance')
    model.fit(X_train, y_train)
    
    # Step 5: Evaluate
    print("\n[5/6] Evaluating model...")
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Step 6: Save model
    print("\n[6/6] Saving model...")
    os.makedirs('models', exist_ok=True)
    model.save('models/knn_model.pkl')
    print("   Model saved to models/knn_model.pkl")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    return model

if __name__ == "__main__":
    # For initial testing, use subsets (KNN is SLOW!)
    # Remove these limits for final training
    train_knn(
        n_neighbors=5,
        train_subset=10000,  # Use 10k samples for faster testing
        test_subset=2000     # Use 2k samples for faster testing
    )
    
    # For full training (will take longer):
    # train_knn(n_neighbors=5, train_subset=None, test_subset=None)

