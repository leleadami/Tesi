"""
LSTM Autoencoder per forecasting di serie temporali

Architettura:
- Encoder: LSTM che comprime la sequenza di input in una rappresentazione bottleneck
- Decoder: LSTM che ricostruisce l'input dal bottleneck
- Training: Minimizza errore di ricostruzione (MSE tra input e ricostruito)
- Forecasting: Usa il bottleneck come feature per la predizione

Riferimento:
Malhotra et al., "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection", 2016
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder per forecasting di serie temporali

    Fase di training: Impara a ricostruire l'input (pre-training)
    Fase di inference: Usa il bottleneck per la predizione (fine-tuning)

    Args:
        input_size: Numero di feature in input
        hidden_size: Dimensione hidden per encoder/decoder LSTM
        bottleneck_size: Dimensione della rappresentazione compressa
        num_layers: Numero di layer LSTM
        dropout: Tasso di dropout
        output_size: Numero di feature in output (default: 1 per forecasting)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        bottleneck_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        use_skip_connections: bool = False  # NUOVO: skip connections
    ):
        super(LSTMAutoencoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_skip_connections = use_skip_connections

        # ENCODER: comprime la sequenza nel bottleneck
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Layer bottleneck: comprime lo stato nascosto dell'encoder
        # Riduce dimensionalità da hidden_size a bottleneck_size
        self.encoder_fc = nn.Linear(hidden_size, bottleneck_size)

        # DECODER: ricostruisce la sequenza originale dal bottleneck
        # Prima espande il bottleneck a hidden_size per il decoder LSTM
        self.decoder_fc = nn.Linear(bottleneck_size, hidden_size)

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Layer di output per ricostruzione: mappa hidden_size a input_size
        self.reconstruction_fc = nn.Linear(hidden_size, input_size)

        # FORECASTING HEAD: usa la rappresentazione compressa per predire il futuro
        # Con skip connections: concatena bottleneck + hidden state (più info)
        # Senza skip: usa solo bottleneck (compresso)
        forecast_input_size = bottleneck_size + hidden_size if use_skip_connections else bottleneck_size
        self.forecast_fc = nn.Sequential(
            nn.Linear(forecast_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        """
        Codifica la sequenza di input in una rappresentazione bottleneck compressa

        Args:
            x: Input (batch_size, seq_len, input_size)

        Returns:
            bottleneck: Rappresentazione compressa (batch_size, bottleneck_size)
            last_hidden: Ultimo hidden state (batch_size, hidden_size) - per skip connections
        """
        # Encoder LSTM: processa l'intera sequenza
        encoder_out, (hidden, cell) = self.encoder(x)

        # Usa l'ultimo hidden state come rappresentazione compatta della sequenza
        # Cattura l'informazione dell'intera sequenza in un vettore
        last_hidden = hidden[-1]  # (batch_size, hidden_size)

        # Comprimi ulteriormente nel bottleneck tramite proiezione lineare
        bottleneck = self.encoder_fc(last_hidden)  # (batch_size, bottleneck_size)

        return bottleneck, last_hidden

    def decode(self, bottleneck, seq_len):
        """
        Decodifica il bottleneck per ricostruire la sequenza di input originale

        Args:
            bottleneck: Rappresentazione compressa (batch_size, bottleneck_size)
            seq_len: Lunghezza della sequenza da ricostruire

        Returns:
            reconstruction: Sequenza ricostruita (batch_size, seq_len, input_size)
        """
        batch_size = bottleneck.size(0)

        # Espandi il bottleneck a hidden_size per il decoder
        decoder_hidden = self.decoder_fc(bottleneck)  # (batch_size, hidden_size)

        # Ripeti il vettore espanso per ogni timestep della sequenza
        decoder_input = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # Risultato: (batch_size, seq_len, hidden_size)

        # Decoder LSTM: genera la sequenza dal vettore ripetuto
        decoder_out, _ = self.decoder(decoder_input)
        # Output: (batch_size, seq_len, hidden_size)

        # Proietta da hidden_size a input_size per ricostruire l'input originale
        reconstruction = self.reconstruction_fc(decoder_out)
        # Risultato: (batch_size, seq_len, input_size)

        return reconstruction

    def forward(self, x, mode='forecast'):
        """
        Forward pass (passata in avanti della rete)

        Args:
            x: Input (batch_size, seq_len, input_size)
            mode: 'forecast' per predizione (default), 'reconstruction' per pre-training

        Returns:
            - Se mode='forecast': predizione per il prossimo timestep
            - Se mode='reconstruction': input ricostruito
        """
        # Codifica: comprime la sequenza nel bottleneck + restituisce hidden state
        bottleneck, last_hidden = self.encode(x)

        if mode == 'reconstruction':
            # Modalità ricostruzione: usata durante pre-training
            # Obiettivo: minimizzare MSE tra input e ricostruito
            seq_len = x.size(1)
            reconstruction = self.decode(bottleneck, seq_len)
            return reconstruction

        elif mode == 'forecast':
            # Modalità forecasting: usata durante fine-tuning e inference
            # Con skip connections: concatena bottleneck + hidden state
            # Senza skip: usa solo bottleneck
            if self.use_skip_connections:
                forecast_input = torch.cat([bottleneck, last_hidden], dim=1)
            else:
                forecast_input = bottleneck

            prediction = self.forecast_fc(forecast_input)
            return prediction.squeeze(-1)

        else:
            raise ValueError(f"Modalità sconosciuta: {mode}. Usa 'reconstruction' o 'forecast'.")

    def get_model_size(self) -> int:
        """Restituisce il numero totale di parametri"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
