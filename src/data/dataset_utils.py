from torchvision import datasets, transforms
import torch
from src.config import EMNIST_SPLIT

def _emnist_fix_orientation(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    EMNIST raw images come rotated 90° counter-clockwise and flipped horizontally.
    To fix them to upright orientation:
    1. First flip horizontally (undoes the horizontal flip)
    2. Then rotate 90° counter-clockwise (matches the original rotation)
    This makes the dataset upright from the start, so no fix needed during inference.
    img_tensor: shape [1, H, W] after ToTensor()
    """
    # Undo horizontal flip first, then rotate 90° counter-clockwise
    # flip(2) = horizontal flip, rot90(k=1) = rotate 90° counter-clockwise
    return img_tensor.flip(2).rot90(k=1, dims=[1, 2])

def get_emnist_datasets(data_dir="data/raw", apply_orientation_fix=True):
    """
    Get EMNIST datasets.
    apply_orientation_fix: If True, applies EMNIST orientation fix (rotate 90° CCW, flip H).
                          If False, uses images as-is (for matching user input orientation).
    """
    if apply_orientation_fix:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(_emnist_fix_orientation),
        ])
    else:
        # No orientation fix - use images in their normal orientation
        transform = transforms.Compose([
            transforms.ToTensor(),
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
    """
    Try to build a human-readable label list.
    EMNIST doesn't always provide a simple list like CIFAR.
    We'll use a fallback unless you build/verify a mapping.
    """
    # You can inspect unique labels and verify mapping later.
    # For now, return placeholder; update once you confirm order.
    from src.config import LABELS_FALLBACK
    return LABELS_FALLBACK
