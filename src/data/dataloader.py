"""
Dataset PyTorch e utilità DataLoader per forecasting di serie temporali
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, List


# Dataset PyTorch per sequenze temporali:
# - Converte array numpy in tensori
# - Restituisce (X, y) per DataLoader
# - Permette di applicare trasformazioni opzionali sulle sequenze

class TimeSeriesDataset(Dataset):
    """
    Dataset PyTorch per forecasting di serie temporali

    Args:
        X: Sequenze di input (N, sequence_length, n_features)
        y: Valori target (N,)
        transform: Trasformazione opzionale da applicare
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


# Crea DataLoader PyTorch per sequenze temporali:
# - Converte dataset numpy in Tensori tramite TimeSeriesDataset
# - Restituisce train_loader e val_loader pronti per il training/valutazione
# - Supporta batching, shuffle e caricamento parallelo

def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Crea DataLoader per training e validation

    Args:
        X_train: Sequenze di training (N, seq_len, features)
        y_train: Target di training (N,)
        X_val: Sequenze di validation
        y_val: Target di validation
        batch_size: Dimensione batch
        num_workers: Numero di worker per caricamento dati
        shuffle_train: Se mescolare i dati di training

    Returns:
        Tupla di (train_loader, val_loader)
    """
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    # pin_memory: attiva solo se GPU disponibile (evita warning su CPU)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader # non mescolato



# DataLoader PyTorch per il test:
# - Converte sequenze di test in Tensori
# - Restituisce batch pronti per inferenza/valutazione
# - Non mescola i dati per preservare l’ordine temporale

def get_test_dataloader(
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0
) -> DataLoader:
    """
    Crea DataLoader per test

    Args:
        X_test: Sequenze di test
        y_test: Target di test
        batch_size: Dimensione batch
        num_workers: Numero di worker

    Returns:
        DataLoader per dati di test
    """
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # pin_memory: attiva solo se GPU disponibile (evita warning su CPU)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return test_loader
