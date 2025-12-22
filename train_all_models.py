#!/usr/bin/env python3
"""
Optimized training script for M2 Pro MacBook.
Uses parallelization, faster models, PCA, and Metal GPU acceleration.
"""
import sys
import os
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from joblib import dump
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from src.data.dataset_utils import get_emnist_datasets
from src.models.knearest_model import KNearestModel
from src.models.cnn_model import SimpleCNN
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

os.makedirs("models", exist_ok=True)

print("="*70)
print("Optimized Training for M2 Pro MacBook")
print("="*70)
print("Features: Parallel processing, PCA, HistGradientBoosting, Metal GPU")
print("="*70)

# Load datasets WITHOUT orientation fix - user's images are in normal orientation
print("\n[1/6] Loading EMNIST datasets (without orientation fix)...")
start_time = time.time()
train_ds, test_ds = get_emnist_datasets(apply_orientation_fix=False)
load_time = time.time() - start_time
print(f"   ✓ Loaded in {load_time:.1f}s")
print(f"   Full dataset: Train={len(train_ds)}, Test={len(test_ds)}")

# Use larger subsets for better accuracy
TRAIN_SIZE = 50000  # Increased for better accuracy
TEST_SIZE = 10000   # Increased for better evaluation

# Prepare data for sklearn models
print(f"\n[2/6] Preparing data ({TRAIN_SIZE} train, {TEST_SIZE} test)...")
start_time = time.time()
train_indices = np.random.choice(len(train_ds), min(TRAIN_SIZE, len(train_ds)), replace=False)
test_indices = np.random.choice(len(test_ds), min(TEST_SIZE, len(test_ds)), replace=False)

print("   Processing training data...")
X_train_sklearn = []
y_train_sklearn = []
for i, idx in enumerate(train_indices):
    if (i + 1) % 5000 == 0:
        print(f"      {i + 1}/{len(train_indices)} samples...")
    img, label = train_ds[idx]
    X_train_sklearn.append(img.numpy().flatten())
    y_train_sklearn.append(label)
X_train_sklearn = np.array(X_train_sklearn, dtype=np.float32)
y_train_sklearn = np.array(y_train_sklearn)

print("   Processing test data...")
X_test_sklearn = []
y_test_sklearn = []
for i, idx in enumerate(test_indices):
    if (i + 1) % 2000 == 0:
        print(f"      {i + 1}/{len(test_indices)} samples...")
    img, label = test_ds[idx]
    X_test_sklearn.append(img.numpy().flatten())
    y_test_sklearn.append(label)
X_test_sklearn = np.array(X_test_sklearn, dtype=np.float32)
y_test_sklearn = np.array(y_test_sklearn)

prep_time = time.time() - start_time
print(f"   ✓ Prepared in {prep_time:.1f}s")
print(f"   Shape: Train={X_train_sklearn.shape}, Test={X_test_sklearn.shape}")

# Apply PCA for dimensionality reduction (huge speedup for sklearn models)
print("\n[3/6] Applying PCA for dimensionality reduction...")
start_time = time.time()
pca = PCA(n_components=100, random_state=42)  # Reduce from 784 to 100 dimensions
X_train_pca = pca.fit_transform(X_train_sklearn)
X_test_pca = pca.transform(X_test_sklearn)
pca_time = time.time() - start_time
print(f"   ✓ PCA applied in {pca_time:.1f}s")
print(f"   Reduced from 784 to 100 dimensions (87% reduction)")
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
dump(pca, "models/pca_transformer.pkl")
print("   ✓ Saved PCA transformer: models/pca_transformer.pkl")

# Train LinearSVM (much faster than RBF kernel SVM)
print("\n[4/6] Training LinearSVM (optimized for M2 Pro)...")
start_time = time.time()
svm = LinearSVC(C=10.0, max_iter=5000, random_state=42, dual=False, verbose=1)  # Increased C and iterations for better accuracy
svm.fit(X_train_pca, y_train_sklearn)
train_time = time.time() - start_time
print(f"   ✓ Training complete in {train_time:.1f}s")
print("   Evaluating on test set...")
preds = svm.predict(X_test_pca)
acc = accuracy_score(y_test_sklearn, preds)
print(f"   Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
dump(svm, "models/svm_emnist.joblib")
dump(pca, "models/svm_pca.pkl")  # Save PCA for inference
dump(pca, "models/pca_transformer.pkl")  # Also save as general PCA
print("   ✓ Saved: models/svm_emnist.joblib + models/svm_pca.pkl")

# Train KNearest (already has n_jobs=-1, but use PCA for speed)
print("\n[5/6] Training KNearest (with parallelization)...")
start_time = time.time()
# Use larger subset for better accuracy
knn_train_size = min(30000, len(X_train_pca))
knn_train_indices = np.random.choice(len(X_train_pca), knn_train_size, replace=False)
X_train_knn = X_train_pca[knn_train_indices]
y_train_knn = y_train_sklearn[knn_train_indices]

knn = KNearestModel(n_neighbors=7, weights='distance')  # Increased neighbors for better accuracy
knn.fit(X_train_knn, y_train_knn)
train_time = time.time() - start_time
print(f"   ✓ Training complete in {train_time:.1f}s (on {knn_train_size} samples)")
print("   Evaluating on test set...")
preds = knn.predict(X_test_pca)
acc = accuracy_score(y_test_sklearn, preds)
print(f"   Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
knn.save("models/knn_model.pkl")
dump(pca, "models/knn_pca.pkl")  # Save PCA for inference
dump(pca, "models/pca_transformer.pkl")  # Also save as general PCA
print("   ✓ Saved: models/knn_model.pkl + models/knn_pca.pkl")

# Train HistGradientBoosting (MUCH faster than GradientBoostingClassifier)
print("\n[6/7] Training HistGradientBoosting (optimized, multithreaded)...")
start_time = time.time()
gb = HistGradientBoostingClassifier(
    max_iter=200,  # Increased iterations
    learning_rate=0.05,  # Lower learning rate for better convergence
    max_depth=15,  # Increased depth
    min_samples_leaf=5,  # Better regularization
    random_state=42,
    verbose=1
)
gb.fit(X_train_pca, y_train_sklearn)
train_time = time.time() - start_time
print(f"   ✓ Training complete in {train_time:.1f}s")
print("   Evaluating on test set...")
preds = gb.predict(X_test_pca)
acc = accuracy_score(y_test_sklearn, preds)
print(f"   Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
dump(gb, "models/gradient_model.pkl")
dump(pca, "models/gradient_pca.pkl")  # Save PCA for inference
dump(pca, "models/pca_transformer.pkl")  # Also save as general PCA
print("   ✓ Saved: models/gradient_model.pkl + models/gradient_pca.pkl")

# Train CNN with Metal GPU acceleration (MPS)
print("\n[7/7] Training CNN with Metal GPU acceleration...")
# Check for MPS (Metal Performance Shaders) on Apple Silicon
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"   ✓ Using Metal GPU (MPS) - M2 Pro acceleration enabled!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"   Using CUDA GPU")
else:
    device = torch.device("cpu")
    print(f"   Using CPU (no GPU acceleration)")

train_subset = Subset(train_ds, train_indices[:30000])  # Use 30k for CNN (better accuracy)
test_subset = Subset(test_ds, test_indices[:5000])      # Use 5k for CNN

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=0)

from src.config import NUM_CLASSES
model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for 10 epochs (better accuracy with GPU)
print(f"   Training on {len(train_subset)} samples, {len(train_loader)} batches per epoch...")
model.train()
start_time = time.time()
for epoch in range(10):
    epoch_start = time.time()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batch_count += 1
        
        # Progress update every 50 batches
        if (batch_idx + 1) % 50 == 0:
            current_acc = correct / total
            print(f"      Batch {batch_idx + 1}/{len(train_loader)} - Loss: {running_loss/batch_count:.4f}, Acc: {current_acc:.4f}", end='\r')
    
    train_acc = correct / total
    epoch_time = time.time() - epoch_start
    print(f"   Epoch {epoch+1}/10 - Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.4f} ({epoch_time:.1f}s)")

train_time = time.time() - start_time
print(f"   ✓ Training complete in {train_time:.1f}s")

# Evaluate
print("   Evaluating on test set...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
acc = correct / total
print(f"   Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

torch.save(model.state_dict(), "models/cnn_mnist.pth")
print("   ✓ Saved: models/cnn_mnist.pth")

total_time = time.time() - start_time
print("\n" + "="*70)
print("All models trained successfully!")
print(f"Total training time: {total_time/60:.1f} minutes")
print("="*70)
print("\nOptimizations applied:")
print("  ✓ PCA dimensionality reduction (784 → 100)")
print("  ✓ LinearSVM (faster than RBF kernel)")
print("  ✓ HistGradientBoosting (multithreaded, faster than GradientBoosting)")
print("  ✓ KNN with n_jobs=-1 (parallelized)")
print("  ✓ CNN with Metal GPU (MPS) acceleration")
print("="*70)
