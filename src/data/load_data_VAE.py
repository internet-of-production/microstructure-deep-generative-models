"""Module with function to get PyTorch DataLoaders for training and testing."""

# Import standard library dependencies
from typing import Tuple

# Import third party dependencies
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def get_data_loaders(
    path,
    train_size=0.5,
    batch_size=32,
    shuffle=False
) -> Tuple[DataLoader, DataLoader]:
    """Get torch DataLoaders for training and testing.

    Parameters
    ----------
    path : _type_
        Path to folder with subfolder(s) containing image data. Subfolder
        name(s) are used as labels.
    train_size : float, optional
        Percentage of data used for training, defaults to 0.5.
    batch_size : int, optional
        Number of images per batch, defaults to 32
    shuffle : bool, optional
        Defines whether data is shuffled, defaults to False.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        The Dataloaders.
    """

    # Define transformations applied to images
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # Load dataset from image folder
    dataset = datasets.ImageFolder(path, transform)

    # Split dataset into train and test datasets
    dataset_length = len(dataset)
    train_length = int(dataset_length * train_size)
    lengths = [train_length, dataset_length - train_length]
    dataset_train, dataset_test = random_split(dataset, lengths)

    # Instantiate DataLoaders
    data_loader_train = DataLoader(dataset_train, batch_size, shuffle)
    data_loader_test = DataLoader(dataset_test, batch_size, shuffle)

    return data_loader_train, data_loader_test


def get_data_loader(
    path: str,
    batch_size: int = 32,
    shuffle: bool = False
) -> DataLoader:
    """Get torch DataLoader.

    Parameters
    ----------
    path : _type_
        Path to folder with subfolder(s) containing image data. Subfolder
        name(s) are used as labels.
    batch_size : int, optional
        Number of images per batch, defaults to 32
    shuffle : bool, optional
        Defines whether data is shuffled, defaults to False.

    Returns
    -------
    DataLoader
        The Dataloader.
    """
    # Define transformations applied to images
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # Load dataset from image folder
    dataset = datasets.ImageFolder(path, transform)

    return DataLoader(dataset, batch_size, shuffle)
