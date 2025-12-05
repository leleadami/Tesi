"""
LSTM Encoder-Decoder con Meccanismo di Attenzione per forecasting di serie temporali

Architettura:
- Encoder: LSTM che processa la sequenza di input e produce stati nascosti
- Attention: Calcola una somma pesata degli stati nascosti dell'encoder
- Decoder: LSTM che usa il context dell'attenzione per generare la predizione

Riferimento:
Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate", ICLR 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Meccanismo di attenzione stile Bahdanau

    Calcola i pesi di attenzione sugli stati nascosti dell'encoder e restituisce
    un context vector come somma pesata degli output dell'encoder

    Args:
        hidden_size: Dimensionalità degli stati nascosti
    """

    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()

        # Proiezioni lineari per calcolo attention scores (Bahdanau)
        self.W_encoder = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_decoder = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Calcola i pesi di attenzione e il context vector

        Args:
            encoder_outputs: Stati nascosti encoder (batch_size, seq_len, hidden_size)
            decoder_hidden: Stato nascosto corrente del decoder (batch_size, hidden_size)

        Returns:
            context: Somma pesata degli output dell'encoder (batch_size, hidden_size)
            attention_weights: Distribuzione dell'attenzione (batch_size, seq_len)
        """
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        # decoder_hidden: (batch_size, hidden_size)

        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # Espansione decoder_hidden per matching dimensionale
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Calcolo energy via formula Bahdanau: tanh(W_e·encoder + W_d·decoder)
        energy = torch.tanh(
            self.W_encoder(encoder_outputs) + self.W_decoder(decoder_hidden)
        )

        # Proiezione energy a scalari: v^T · energy
        attention_scores = self.v(energy).squeeze(-1)

        # Normalizzazione scores a distribuzione di probabilità (somma = 1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Context vector: somma pesata degli output encoder
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            encoder_outputs                  # (batch, seq_len, hidden)
        ).squeeze(1)                         # (batch, hidden)

        return context, attention_weights


class LSTMAttention(nn.Module):
    """
    LSTM Encoder-Decoder con Attenzione per forecasting sequence-to-sequence

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
        super(LSTMAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Encoder: processa la sequenza di input
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Meccanismo di attenzione
        self.attention = AttentionLayer(hidden_size)

        # Decoder: genera l'output usando il context dall'attenzione
        self.decoder = nn.LSTM(
            input_size=1,  # Decoder riceve input semplice (valore scalare)
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Layer fully connected per combinare output del decoder e context dell'attenzione
        # Input: concatenazione di decoder_output (hidden_size) e context (hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)
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

        # ENCODER: processa tutta la sequenza di input
        # encoder_outputs: (batch_size, seq_len, hidden_size) - tutti gli stati nascosti
        # hidden: (num_layers, batch_size, hidden_size) - stato nascosto finale
        # cell: (num_layers, batch_size, hidden_size) - stato della cella finale
        encoder_outputs, (hidden, cell) = self.encoder(x)

        # DECODER: usa il context vector dall'encoder per inizializzare
        # Decoder input: tensor di zeri (batch_size, 1, 1) - input fittizio
        decoder_input = torch.zeros(batch_size, 1, 1).to(x.device)

        # Forward del decoder usando stati dall'encoder come inizializzazione
        decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (hidden, cell))

        # Rimuovi dimensione temporale: decoder_output da (batch_size, 1, hidden_size) a (batch_size, hidden_size)
        decoder_output = decoder_output.squeeze(1)

        # ATTENTION: calcola context vector basato sugli output dell'encoder
        # context: (batch_size, hidden_size) - somma pesata degli stati encoder
        # attention_weights: (batch_size, seq_len) - pesi di attenzione per ogni timestep
        context, attention_weights = self.attention(encoder_outputs, decoder_output)

        # Combina output del decoder con context dell'attenzione tramite concatenazione
        # Risultato: (batch_size, hidden_size * 2)
        combined = torch.cat([decoder_output, context], dim=1)
        combined = self.dropout(combined)

        # Output finale attraverso layer fully connected
        out = self.fc(combined)  # (batch_size, output_size)

        return out.squeeze(-1)  # (batch_size,)

    def get_model_size(self) -> int:
        """Calcola numero totale di parametri addestrabili"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_weights(self, x):
        """
        Ottiene i pesi di attenzione per visualizzazione

        Utile per interpretare quali timestep della sequenza input
        il modello considera più importanti per la predizione.

        Args:
            x: Tensore di input (batch_size, sequence_length, input_size)

        Returns:
            attention_weights: Pesi di attenzione (batch_size, seq_len)
                              Valori tra 0 e 1, somma = 1 per ogni batch
        """
        batch_size = x.size(0)

        # Encoder: processa la sequenza
        encoder_outputs, (hidden, cell) = self.encoder(x)

        # Decoder: genera stato per calcolare attenzione
        decoder_input = torch.zeros(batch_size, 1, 1).to(x.device)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        decoder_output = decoder_output.squeeze(1)

        # Attention: calcola solo i pesi (ignora context vector)
        _, attention_weights = self.attention(encoder_outputs, decoder_output)

        return attention_weights
