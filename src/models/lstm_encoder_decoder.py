"""
Modello LSTM Encoder-Decoder per forecasting di serie temporali

Architettura:
- Encoder: LSTM che processa la sequenza di input e produce un context vector
- Decoder: LSTM che usa il context per generare la predizione

Riferimento:
Sutskever et al., "Sequence to Sequence Learning with Neural Networks", NeurIPS 2014
"""

import torch
import torch.nn as nn


class LSTMEncoderDecoder(nn.Module):
    """
    LSTM Encoder-Decoder per forecasting sequence-to-sequence

    Args:
        input_size: Numero di feature in input
        hidden_size: Dimensione hidden sia per encoder che decoder LSTM
        num_layers: Numero di layer LSTM sia per encoder che decoder
        dropout: Tasso di dropout
        output_size: Numero di feature in output (default: 1 per forecast singolo step)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super(LSTMEncoderDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Encoder: estrae rappresentazione dalla sequenza input
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Decoder: genera output dal context vector
        self.decoder = nn.LSTM(
            input_size=1,  # Input scalare (zero-input o valore condizionato)
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Proiezione finale e regolarizzazione
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass (passata in avanti della rete)

        Args:
            x: Tensore di input (batch_size, sequence_length, input_size)

        Returns:
            Predizione output (batch_size,) per forecast singolo step
        """
        batch_size = x.size(0)

        # ENCODER: estrae context dalla sequenza
        # encoder_output: (batch, seq_len, hidden_size)
        # hidden, cell: stati finali (num_layers, batch, hidden_size)
        encoder_output, (hidden, cell) = self.encoder(x)

        # DECODER: genera predizione dal context
        # Input: zero-tensor (single-step forecasting non richiede input reale)
        decoder_input = torch.zeros(batch_size, 1, 1).to(x.device)

        # Forward decoder con inizializzazione da encoder
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

        # Proiezione finale
        decoder_output = decoder_output.squeeze(1)  # (batch, hidden_size)
        decoder_output = self.dropout(decoder_output)
        out = self.fc(decoder_output)

        return out.squeeze(-1)

    def get_model_size(self) -> int:
        """Calcola numero totale di parametri addestrabili"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
