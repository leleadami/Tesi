"""
Modello LSTM Vanilla per forecasting di serie temporali (baseline)
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMModel(nn.Module):
    """
    LSTM Vanilla per forecasting di serie temporali

    Architettura:
        Input → Layer LSTM → Dropout → Fully Connected → Output

    Args:
        input_size: Numero di feature in input
        hidden_size: Numero di unità nascoste nell'LSTM
        num_layers: Numero di layer LSTM impilati
        dropout: Tasso di dropout (applicato tra i layer LSTM)
        bidirectional: Se usare LSTM bidirezionale
        output_size: Numero di valori in output (default 1 per forecast singolo step)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1
    ):
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = output_size

        # LSTM layer (dropout automatico tra layer se num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Fully connected: adatta dimensionalità per bidirezionalità
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)

        # Dropout post-LSTM per regolarizzazione
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (passata in avanti della rete)

        Args:
            x: Tensore di input con shape (batch_size, sequence_length, input_size)

        Returns:
            Tensore di output con shape (batch_size, output_size)
        """
        # Forward attraverso LSTM
        # lstm_out: (batch, seq_len, hidden_size × num_directions)
        # h_n, c_n: stati finali (num_layers × num_directions, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Estrazione ultimo timestep per predizione
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size × num_directions)

        # Regolarizzazione e proiezione finale
        last_output = self.dropout(last_output)
        out = self.fc(last_output)

        return out.squeeze(-1) if self.output_size == 1 else out

    def get_model_size(self) -> int:
        """Calcola numero totale di parametri addestrabili"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_lstm_model(
    input_size: int,
    config: dict | None = None
) -> LSTMModel:
    """
    Factory function per creare un modello LSTM da configurazione

    Args:
        input_size: Numero di feature in input
        config: Dizionario di configurazione con iperparametri

    Returns:
        Modello LSTM inizializzato
    """
    if config is None:
        config = {}

    model = LSTMModel(
        input_size=input_size,
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.2),
        bidirectional=config.get('bidirectional', False),
        output_size=config.get('output_size', 1)
    )

    return model