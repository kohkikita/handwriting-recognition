from torchvision import datasets, transforms
import torch

NUM_CLASSES = 47  # EMNIST balanced


def _emnist_fix(img):
    """
    Fix EMNIST orientation (images are rotated/flipped by default)
    """
    return torch.rot90(img, k=1, dims=[1, 2]).flip(2)


def get_emnist_datasets(data_dir="data/raw"):
    """
    Returns EMNIST Balanced train and test datasets
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_emnist_fix),
    ])

    train_dataset = datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset
