import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from tqdm import tqdm

from src.data.dataset_utils import get_emnist_datasets
from src.models.cnn_model import SimpleCNN

MODEL_PATH = "models/cnn_emnist.pth"

def main():
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = get_emnist_datasets()

    # Validation split (e.g., 10% of train)
    val_size = int(0.1 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_subset, val_subset = random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs = 8

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_acc = correct / total
        train_loss = loss_sum / total

        # Validate
        model.eval()
        vtotal, vcorrect = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                vcorrect += (preds == y).sum().item()
                vtotal += x.size(0)

        val_acc = vcorrect / vtotal
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best:", MODEL_PATH)

    # Test (load best)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    ttotal, tcorrect = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            tcorrect += (preds == y).sum().item()
            ttotal += x.size(0)

    print("Test Accuracy:", tcorrect / ttotal)

if __name__ == "__main__":
    main()
