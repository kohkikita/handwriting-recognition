import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from src.data.dataset_utils import get_emnist_datasets
from src.models.cnn_model import SimpleCNN
from src.config import NUM_CLASSES

MODEL_PATH = "models/cnn_mnist.pth"
BATCH_SIZE = 64
EPOCHS = 3  # Reduced for faster training
LEARNING_RATE = 0.001

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds, all_labels

def main():
    os.makedirs("models", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading EMNIST datasets...")
    train_ds, test_ds = get_emnist_datasets()
    
    # Use subset for faster training
    print(f"Full dataset: Train={len(train_ds)}, Test={len(test_ds)}")
    print("Using subset for faster training...")
    
    # Create subset datasets
    train_indices = torch.randperm(len(train_ds))[:10000]  # Use 10k samples
    test_indices = torch.randperm(len(test_ds))[:2000]     # Use 2k samples
    
    train_subset = torch.utils.data.Subset(train_ds, train_indices)
    test_subset = torch.utils.data.Subset(test_ds, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Training on {len(train_subset)} samples, testing on {len(test_subset)} samples")
    
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nTraining CNN for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    print("\nEvaluating on test set...")
    test_acc, preds, labels = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(classification_report(labels, preds, digits=4))
    
    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()

