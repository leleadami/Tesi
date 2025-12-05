"""
Utilità per Cross Validation su Serie Temporali

Implementa sliding window cross validation per forecasting di serie temporali,
evitando temporal leakage assicurando che i dati di training siano sempre prima dei dati di test.
"""

import numpy as np
from typing import Iterator, Tuple


class TimeSeriesSlidingWindow:
    """
    Sliding window cross validation per serie temporali.

    Garantisce ordinamento temporale: training sempre prima del testing.
    Mantiene dimensione train/test costante tra i fold.

    Parametri:
    ----------
    train_size : float
        Proporzione di dati per finestra di training (es. 0.4 = 40%)
    test_size : float
        Proporzione di dati per finestra di test (es. 0.2 = 20%)
    step : float, opzionale
        Offset tra fold consecutivi (es. 0.1 = 10%)
        Se non fornito, sarà calcolato da n_splits
    n_splits : int, opzionale
        Numero di fold da generare. Se fornito, step è calcolato automaticamente.
        Ha priorità sul parametro step.

    Esempio:
    --------
    >>> cv = TimeSeriesSlidingWindow(train_size=0.4, test_size=0.2, step=0.1)
    >>> for train_idx, test_idx in cv.split(X):
    ...     # Fold 1: train [0-40%], test [40-60%]
    ...     # Fold 2: train [10-50%], test [50-70%]
    ...     # Fold 3: train [20-60%], test [60-80%]
    ...     # ecc.

    >>> # Alternativa: specifica numero di split direttamente
    >>> cv = TimeSeriesSlidingWindow(train_size=0.4, test_size=0.2, n_splits=5)
    """

    def __init__(self, train_size: float = 0.4, test_size: float = 0.2,
                 step: float | None = None, n_splits: int | None = None):
        if train_size <= 0 or train_size >= 1:
            raise ValueError("train_size must be between 0 and 1")
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        if train_size + test_size > 1:
            raise ValueError("train_size + test_size cannot exceed 1")

        self.train_size = train_size
        self.test_size = test_size
        self.n_splits_target = n_splits

        # Calcolo step da n_splits (se fornito) o uso valore esplicito
        if n_splits is not None:
            if n_splits < 1:
                raise ValueError("n_splits deve essere >= 1")
            available_space = 1.0 - train_size - test_size
            if n_splits == 1:
                self.step = available_space
            else:
                self.step = available_space / (n_splits - 1)
        elif step is not None:
            if step <= 0 or step >= 1:
                raise ValueError("step deve essere tra 0 e 1")
            self.step = step
        else:
            self.step = 0.1  # Default step

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera indici train/test per sliding window splits.

        Parametri:
        ----------
        X : array-like, shape (n_samples, ...)
            Dati di training

        Restituisce:
        ------------
        train_idx : ndarray
            Indici del training set
        test_idx : ndarray
            Indici del test set
        """
        n_samples = len(X)
        start = 0.0

        while True:
            # Calcolo finestre train e test
            train_start = int(start * n_samples)
            train_end = int((start + self.train_size) * n_samples)
            test_start = train_end
            test_end = int((start + self.train_size + self.test_size) * n_samples)

            # Terminazione se finestra supera dimensione dataset
            if test_end > n_samples:
                break

            # Generazione indici fold
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

            # Scorrimento finestra
            start += self.step

    def get_n_splits(self, X: np.ndarray) -> int:
        """
        Ottiene il numero di split che saranno generati.

        Parametri:
        ----------
        X : array-like
            Dati di training

        Returns:
        --------
        n_splits : int
            Numero di fold
        """
        n_samples = len(X)
        start = 0.0
        n_splits = 0

        while True:
            test_end = (start + self.train_size + self.test_size) * n_samples
            if test_end > n_samples:
                break
            n_splits += 1
            start += self.step

        return n_splits


def print_split_info(cv, n_samples: int):
    """
    Stampa informazioni sugli sliding window splits.

    Parametri:
    ----------
    cv : TimeSeriesSlidingWindow
        Splitter per cross-validation
    n_samples : int
        Numero totale di campioni
    """
    print(f"\nConfigurazione Sliding Window:")
    print(f"  Dimensione train: {cv.train_size*100:.1f}% ({int(cv.train_size*n_samples)} campioni)")
    print(f"  Dimensione test:  {cv.test_size*100:.1f}% ({int(cv.test_size*n_samples)} campioni)")
    print(f"  Step:             {cv.step*100:.1f}% ({int(cv.step*n_samples)} campioni)")
    print(f"  Campioni totali: {n_samples}")

    # Calcola numero di fold
    n_folds = cv.get_n_splits(np.arange(n_samples))
    print(f"  Numero di fold: {n_folds}")

    print(f"\nDettaglio fold:")
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(np.arange(n_samples)), 1):
        # Calcola percentuali usando indici inizio/fine effettivi
        train_pct_start = (train_idx[0] / n_samples) * 100
        train_pct_end = ((train_idx[-1] + 1) / n_samples) * 100  # +1 perché [-1] è l'ultimo indice
        test_pct_start = (test_idx[0] / n_samples) * 100
        test_pct_end = ((test_idx[-1] + 1) / n_samples) * 100

        print(f"  Fold {fold_idx}: train [{train_pct_start:.1f}-{train_pct_end:.1f}%] ({len(train_idx)} campioni) "
              f"-> test [{test_pct_start:.1f}-{test_pct_end:.1f}%] ({len(test_idx)} campioni)")
