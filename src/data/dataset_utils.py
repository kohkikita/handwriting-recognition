from torchvision import datasets, transforms
import torch
from src.config import EMNIST_SPLIT

def _emnist_fix_orientation(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    EMNIST images are often rotated/flipped depending on source.
    This fix is commonly needed for correct visual orientation.
    img_tensor: shape [1, H, W] after ToTensor()
    """
    # rotate 90 degrees then flip horizontally
    return torch.rot90(img_tensor, k=1, dims=[1, 2]).flip(2)

def get_emnist_datasets(data_dir="data/raw"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_emnist_fix_orientation),
    ])

    train_ds = datasets.EMNIST(
        root=data_dir,
        split=EMNIST_SPLIT,
        train=True,
        download=True,
        transform=transform
    )

    test_ds = datasets.EMNIST(
        root=data_dir,
        split=EMNIST_SPLIT,
        train=False,
        download=True,
        transform=transform
    )

    return train_ds, test_ds

def get_label_list(train_ds) -> list[str]:

    from src.config import LABELS_FALLBACK
    return LABELS_FALLBACK