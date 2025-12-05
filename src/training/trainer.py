"""
Utilità per il training di modelli di forecasting di serie temporali
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import json


class EarlyStopping:
    """
    Early stopping per fermare il training quando la validation loss non migliora

    Monitora una metrica (tipicamente validation loss) e ferma il training se non
    migliora per un numero specificato di epoche (patience).

    Args:
        patience: Numero di epoche da attendere prima di fermare
        min_delta: Variazione minima per qualificare come miglioramento
        mode: 'min' per loss (minore è meglio), 'max' per metriche (maggiore è meglio)
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        # Prima chiamata: inizializza best score
        if self.best_score is None:
            self.best_score = score
            return False

        # Verifica miglioramento score
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """
    Trainer generico per modelli PyTorch

    Gestisce il loop di training completo con:
    - Training e validation per epoca
    - Early stopping automatico
    - Learning rate scheduling
    - Salvataggio checkpoint
    - Gradient clipping

    Args:
        model: Modello PyTorch
        train_loader: DataLoader per training
        val_loader: DataLoader per validation
        optimizer: Ottimizzatore
        device: Device su cui trainare (cpu/cuda)
        checkpoint_dir: Directory dove salvare checkpoint
        experiment_name: Nome dell'esperimento
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
        checkpoint_dir: str = '../results/checkpoints',
        experiment_name: str = 'experiment'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name

        # Creazione directory checkpoint
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss function (MSE per regressione)
        self.criterion = nn.MSELoss()

        # Storia metriche di training
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train_epoch(self, clip_grad: float = 1.0) -> float:
        """
        Trainingper una singola epoca

        Args:
            clip_grad: Norma massima per gradient clipping

        Returns:
            Loss media di training
        """
        self.model.train()
        total_loss = 0.0

        with tqdm(self.train_loader, desc='Training', leave=False, ncols=100,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| [{n_fmt}/{total_fmt}]',
                  ascii=False, colour='cyan') as pbar:
            for batch_idx, (X, y) in enumerate(pbar):
                X, y = X.to(self.device), y.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)

                # Backward pass
                loss.backward()

                # Gradient clipping (previene exploding gradients in RNN)
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

                # Aggiornamento parametri
                self.optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self) -> float:
        """
        Validazione sul validation set

        Returns:
            Loss media di validation
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation', leave=False, ncols=100,
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| [{n_fmt}/{total_fmt}]',
                      ascii=False, colour='green') as pbar:
                for X, y in pbar:
                    X, y = X.to(self.device), y.to(self.device)

                    output = self.model(X)
                    loss = self.criterion(output, y)

                    total_loss += loss.item()
                    # Progress bar senza info extra

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Salva checkpoint del modello

        Args:
            epoch: Epoca corrente
            val_loss: Validation loss
            is_best: Se questo è il miglior modello finora
        """
        # Salvataggio best checkpoint (ottimizzazione spazio)
        if is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'history': self.history
            }

            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Miglior modello salvato (val_loss: {val_loss:.6f})")

    def load_checkpoint(self, checkpoint_path: str):
        """Carica checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['val_loss']

    def train(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        scheduler_patience: int = 5,
        clip_grad: float = 1.0,
        verbose: bool = True
    ) -> Dict:
        """
        Loop di training completo

        Args:
            num_epochs: Numero massimo di epoche
            early_stopping_patience: Patience per early stopping
            scheduler_patience: Patience per learning rate scheduler
            clip_grad: Norma massima per gradient clipping
            verbose: Stampa progresso

        Returns:
            Dizionario con history del training
        """
        # Setup early stopping e scheduler
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
        scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=scheduler_patience
        )

        best_val_loss = float('inf')

        print(f"\n{'='*60}")
        print(f"Training {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Total epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Gradient clipping: {clip_grad}")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            # Trainin per un'epoca
            train_loss = self.train_epoch(clip_grad=clip_grad)

            # Validazione sul validation set
            val_loss = self.validate()

            # Aggiorna learning rate basandosi sulla validation loss
            scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Traccia history delle metriche
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            # Verifica se questo è il miglior modello finora
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # Salva checkpoint (sempre latest, condizionalmente best)
            self.save_checkpoint(epoch, val_loss, is_best)

            # Stampa progresso
            if verbose:
                print(f"Epoch {epoch}/{num_epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.6f}")

            # Verifica early stopping
            if early_stopping(val_loss):
                print(f"\n✓ Early stopping attivato all'epoca {epoch}")
                print(f"✓ Miglior validation loss: {best_val_loss:.6f}")
                break

        # Salva history finale
        history_path = self.checkpoint_dir / f"{self.experiment_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"History saved to: {history_path}")
        print(f"{'='*60}\n")

        return self.history


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera predizioni su un dataset

    Args:
        model: Modello trainato
        data_loader: DataLoader con dati
        device: Device su cui eseguire (cpu/cuda)

    Returns:
        Tupla di (predictions, targets)
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            output = model(X)

            predictions.append(output.cpu().numpy())
            targets.append(y.numpy())

    # Concatena tutti i batch in un unico array
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    return predictions, targets
