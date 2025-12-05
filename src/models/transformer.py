"""
Transformer Vanilla per forecasting di serie temporali

Architettura:
- Positional Encoding: Aggiunge informazione sulla posizione temporale
- Multi-Head Self-Attention: Impara relazioni tra timestep
- Feed-Forward Network: Trasformazioni non lineari
- Layer Normalization: Stabilizza il training
- Global Pooling: Aggrega informazione della sequenza

Riferimento:
Vaswani et al., "Attention is All You Need", NeurIPS 2017
"""

# Possibili miglioramenti futuri:
#   - Causal Masking per rispettare causalità temporale
#   - Strategie di tokenizzazione (Patching / Feature-wise Tokens)
#   - Decomposizione serie temporali & Attention specializzata
#   - Modellazione multi-scala per sequenze lunghe
#   - Feature nel dominio delle frequenze & Spectral Attention
#   - Regolarizzazione, Adaptation & Transfer Learning



import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding per Transformer

    Aggiunge informazione posizionale sinusoidale agli embedding di input.
    Il positional encoding permette al modello di capire l'ordine temporale
    della sequenza, dato che self-attention è invariante alla permutazione.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    dove pos è la posizione nella sequenza e i è la dimensione.

    Args:
        d_model: Dimensione del modello (deve essere pari)
        max_len: Lunghezza massima della sequenza
        dropout: Tasso di dropout
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Crea matrice di positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Applica sin alle dimensioni pari, cos alle dimensioni dispari
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Aggiunge dimensione batch: (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Registra come buffer (non parametro trainabile)

        # Type hint per il type checker (pe è un Tensor, non Module)
        self.pe: torch.Tensor

    def forward(self, x):
        """
        Aggiunge positional encoding all'input

        Args:
            x: Tensore di input (batch_size, seq_len, d_model)

        Returns:
            Tensore con positional encoding aggiunto (stessa shape di input)
        """
        # Somma il positional encoding all'input (broadcasting su dimensione batch)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnableWeightedPooling(nn.Module):
    """
    Learnable Weighted Pooling per aggregare sequenze temporali

    Invece di fare media uniforme (tutti i timestep peso uguale),
    impara PESI DIVERSI per ogni timestep (come LSTM Attention).

    Esempio:
        Input: 24 vettori (uno per ora)
        Pesi appresi: [0.05, 0.08, ..., 0.31]  ← ore recenti più importanti
        Output: somma pesata dei 24 vettori

    Args:
        d_model: Dimensione del modello
    """

    def __init__(self, d_model: int):
        super(LearnableWeightedPooling, self).__init__()

        # Layer lineare che genera 1 score per ogni timestep
        # Input: (batch, seq_len, d_model)
        # Output: (batch, seq_len, 1) - uno score per timestep
        self.attention_weights = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Calcola media pesata con pesi appresi

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, d_model) - aggregato pesato
            weights: (batch_size, seq_len) - pesi di attenzione (per visualizzazione)
        """
        # Calcola score per ogni timestep
        # (batch, seq_len, d_model) → (batch, seq_len, 1)
        scores = self.attention_weights(x)

        # Converti scores in pesi con softmax (somma = 1)
        # (batch, seq_len, 1) → (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)

        # Media pesata: moltiplica ogni timestep per il suo peso
        # (batch, seq_len, d_model) * (batch, seq_len, 1) → (batch, seq_len, d_model)
        weighted = x * weights

        # Somma tutti i timestep pesati
        # (batch, seq_len, d_model) → (batch, d_model)
        output = weighted.sum(dim=1)

        return output, weights.squeeze(-1)  # output, weights per interpretabilità


class TransformerModel(nn.Module):
    """
    Transformer Vanilla per forecasting di serie temporali

    Architettura:
        Input → Proiezione Lineare → Positional Encoding →
        Transformer Encoder (xnum_layers) → Global Pooling → Output

    Args:
        input_size: Numero di feature in input
        d_model: Dimensione del modello (hidden size)
        nhead: Numero di teste di attenzione
        num_layers: Numero di layer transformer encoder
        dim_feedforward: Dimensione della rete feedforward
        dropout: Tasso di dropout
        output_size: Numero di valori in output (default 1 per forecasting)
        use_causal_mask: Se True, applica maschera causale per impedire attention su timestep futuri
                         (default True, raccomandato per time series forecasting)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        output_size: int = 1,
        use_causal_mask: bool = True,
        use_weighted_pooling: bool = False
    ):
        super(TransformerModel, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_causal_mask = use_causal_mask
        self.use_weighted_pooling = use_weighted_pooling

        # Proiezione input: mappa input_size a d_model
        # Necessario perché Transformer opera in spazio d_model dimensionale
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding per aggiungere informazione posizionale
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Singolo layer Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Input shape: (batch, seq, feature) invece di (seq, batch, feature)
        )

        # Stack di layer encoder (ripete encoder_layer per num_layers volte)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Layer di output
        # Scelta tra average pooling (uniforme) e weighted pooling (learnable)
        if use_weighted_pooling:
            self.pooling = LearnableWeightedPooling(d_model)
        else:
            self.pooling = nn.AdaptiveAvgPool1d(1)  # Average pooling (default)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, output_size)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Genera maschera causale triangolare superiore per impedire all'attention
        di guardare timestep futuri.

        La maschera impedisce temporal data leakage durante il training, rendendo
        il modello più realistico per deployment in produzione.

        Args:
            sz: Lunghezza della sequenza

        Returns:
            Maschera (sz, sz) con -inf nella parte triangolare superiore

        Esempio (sz=4):
            [[0,    -inf, -inf, -inf],
             [0,       0, -inf, -inf],
             [0,       0,    0, -inf],
             [0,       0,    0,    0]]

            t=0 può vedere: [t0]           ← Solo presente
            t=1 può vedere: [t0, t1]       ← Passato + presente
            t=2 può vedere: [t0, t1, t2]
            t=3 può vedere: [t0, t1, t2, t3]
        """
        # Crea matrice triangolare superiore (diagonal=1 per escludere diagonale)
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)

        # Converti: 1 → -inf (mascherato), 0 → 0 (visibile)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        return mask

    def forward(self, x):
        """
        Forward pass (passata in avanti della rete)

        Args:
            x: Tensore di input (batch_size, seq_len, input_size)

        Returns:
            Tensore di output (batch_size, output_size) o (batch_size,) se output_size=1
        """
        # Proiezione input: (batch, seq_len, input_size) → (batch, seq_len, d_model)
        x = self.input_projection(x)

        # Aggiunge positional encoding
        x = self.pos_encoder(x)

        # Genera maschera causale se richiesto (evita guardare il futuro)
        mask = None
        if self.use_causal_mask:
            seq_len = x.size(1)
            mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        # Transformer encoder: applica self-attention e feedforward
        # Input/Output: (batch, seq_len, d_model)
        # mask: Impedisce attention su timestep futuri (causalità temporale)
        x = self.transformer_encoder(x, mask=mask)

        # Pooling: aggrega informazione temporale (24 timestep → 1 vettore)
        if self.use_weighted_pooling:
            # Weighted pooling: impara pesi diversi per ogni timestep
            # (batch, seq_len, d_model) → (batch, d_model)
            x, _ = self.pooling(x)  # Ignora i pesi (per ora)
        else:
            # Average pooling: media uniforme di tutti i timestep
            # (batch, seq_len, d_model) → (batch, d_model, seq_len) → (batch, d_model)
            x = x.transpose(1, 2)  # Scambia dimensioni per pooling
            x = self.pooling(x).squeeze(-1)  # Fa media su seq_len

        # Dropout e layer fully connected per output
        x = self.dropout(x)
        out = self.fc_out(x)  # (batch, output_size)

        return out.squeeze(-1) if self.output_size == 1 else out

    def get_model_size(self) -> int:
        """Restituisce il numero totale di parametri addestrabili"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer_model(
    input_size: int,
    config: dict | None = None
) -> TransformerModel:
    """
    Factory function per creare un modello Transformer da configurazione

    Args:
        input_size: Numero di feature in input
        config: Dizionario di configurazione con iperparametri

    Returns:
        Modello Transformer inizializzato
    """
    if config is None:
        config = {}

    model = TransformerModel(
        input_size=input_size,
        d_model=config.get('d_model', 128),
        nhead=config.get('nhead', 8),
        num_layers=config.get('num_layers', 2),
        dim_feedforward=config.get('dim_feedforward', 512),
        dropout=config.get('dropout', 0.2),
        output_size=config.get('output_size', 1),
        use_causal_mask=config.get('use_causal_mask', True),
        use_weighted_pooling=config.get('use_weighted_pooling', False)
    )

    return model
