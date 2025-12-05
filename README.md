# Forecasting Energetico in Edifici con Deep Learning

Progetto di ricerca sulla previsione del consumo energetico in edifici residenziali tramite architetture LSTM e Transformer, utilizzando il dataset CityLearn Challenge 2023.

## Panoramica

Il progetto confronta cinque architetture di reti neurali per la previsione oraria del consumo energetico con finestra temporale di 24 ore. Vengono valutati tre scenari sperimentali: forecasting su singolo edificio, transfer learning cross-building e aggregazione a livello di quartiere.

**Dominio:** Smart Buildings, Energy Management Systems

**Task:** Previsione 1 ora avanti con lookback di 24 ore

**Variabili target:**
- Cooling demand (kWh)
- Solar generation (kW)
- Carbon intensity (kg CO2/kWh)

## Dataset

**CityLearn Challenge 2023** - Dataset benchmark per l'ottimizzazione energetica in edifici residenziali.

- **Edifici:** 3 residenziali in zona climatica temperata
- **Copertura temporale:** Estate 2023, risoluzione oraria
- **Fasi:**
  - Phase 1: 720 ore (training principale)
  - Phase 2 Local: 720 ore (validazione estesa)
  - Phase 2 Online: 2.208 ore (test finale)
- **Feature:** 37 variabili (meteo, encoding temporali, caratteristiche edifici)
- **Fonte:** [CityLearn Challenge 2023](https://www.aicrowd.com/challenges/citylearn-challenge-2023)

## Modelli Implementati

### 1. LSTM Baseline
Architettura LSTM vanilla come riferimento di performance. Usa l'ultimo hidden state per la predizione.

### 2. LSTM Encoder-Decoder
Architettura sequence-to-sequence con fasi di encoding e decoding separate. **Miglior performer** (8.1% miglioramento RMSE vs baseline).

### 3. LSTM Attention
Implementa meccanismo di attenzione di Bahdanau permettendo focus dinamico sui timestep rilevanti (7.5% miglioramento RMSE).

### 4. LSTM Autoencoder
Architettura con bottleneck e training in due fasi (pre-training + fine-tuning). Performance inferiore dovuta alla compressione informativa.

### 5. Transformer
Multi-head self-attention con positional encoding per modellazione time series. Elaborazione parallela dell'intera sequenza.

## Installazione

### Prerequisiti

- Python 3.8+
- PyTorch 2.0+

### Setup Ambiente

```bash
# Clone repository
git clone <repository-url>
cd tesi

# Crea e attiva ambiente virtuale
python3 -m venv .venv
source .venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt

# Installa PyTorch (scegli versione appropriata)
# CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Verifica Installazione

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
```

## Uso Rapido

### Training Singolo Edificio

```bash
# LSTM Encoder-Decoder (consigliato)
python experiments/encoder_decoder/train_encoder_decoder.py --hidden_size 128 --building_id 1

# LSTM Baseline
python experiments/baseline/train_baseline.py --hidden_size 128 --building_id 1

# LSTM Attention
python experiments/attention/train_attention.py --hidden_size 128 --building_id 1

# Transformer
python experiments/transformer/train_transformer.py --d_model 128 --nhead 8 --building_id 1

# LSTM Autoencoder
python experiments/autoencoder/train_autoencoder.py --hidden_size 128 --bottleneck_size 32 --building_id 1
```

### Cross-Building Transfer Learning

```bash
# Addestra su Building 1, testa su Buildings 2+3 (fusi)
python experiments/cross_building/train_cross_building.py \
    --train_building 1 \
    --model encoder_decoder

# Valutazione sistematica tutte combinazioni
for b in 1 2 3; do
    python experiments/cross_building/train_cross_building.py \
        --train_building $b \
        --model encoder_decoder
done
```

### Neighborhood Aggregation

```bash
# Solar generation (consigliato - fisicamente aggregabile)
python experiments/neighborhood/train_neighborhood.py \
    --model encoder_decoder \
    --target solar_generation

# Tutti i target supportati
for target in cooling_demand solar_generation carbon_intensity; do
    python experiments/neighborhood/train_neighborhood.py \
        --model encoder_decoder \
        --target $target
done
```

### Time Series Cross-Validation

```bash
# K-fold CV singolo edificio (default: 5 fold)
python experiments/encoder_decoder/train_kfold_encoder_decoder.py \
    --building_id 1 \
    --target cooling_demand \
    --k_folds 5

# K-fold CV aggregazione quartiere
python experiments/neighborhood/train_kfold_neighborhood.py \
    --model encoder_decoder \
    --target solar_generation \
    --k_folds 5
```

## Risultati

### Performance Singolo Edificio (Phase 1 → Phase 2)

| Modello | RMSE | R² | NMAE | Note |
|---------|------|-----|------|------|
| **Encoder-Decoder** | 0.7110 | 0.8105 | 9.2% | Miglior performer |
| **Attention** | 0.7154 | 0.8082 | 9.5% | Secondo miglior risultato |
| **Baseline** | 0.7454 | 0.7918 | 10.8% | Architettura di riferimento |
| **Transformer** | 0.7523 | 0.7845 | 11.1% | Dataset limitato |
| **Autoencoder** | 0.9825 | 0.5673 | 15.9% | Bottleneck penalizza |

### Generalizzazione Cross-Building

- **R² medio:** 0.6105 (calo 19% rispetto in-building)
- **Miglior combinazione:** B3 → B1+B2 (R² = 0.75)
- **Limitazione:** Eterogeneità architettonica limita transfer learning

### Neighborhood Aggregation

- **R² medio:** 0.8393 (eccellente generalizzazione)
- **Target migliore:** Solar generation (R² = 0.87)
- **Carbon intensity:** R² = 0.60 (variabile a livello rete)

## Aiuto

Tutti gli script di training forniscono help dettagliato:

```bash
python experiments/baseline/train_baseline.py --help
python experiments/cross_building/train_cross_building.py --help
python experiments/neighborhood/train_kfold_neighborhood.py --help
```

## Note Importanti

- **Temporal Leakage:** Usare sempre sliding window CV per time series
- **Normalizzazione:** MinMaxScaler per variabili limitate, StandardScaler per carichi
- **Early Stopping:** Default patience=15 ottimizzato per Phase 1 (720h)
- **PyTorch:** Installare separatamente in base a CPU/CUDA

## Riferimenti

- **Dataset:** [CityLearn Challenge 2023](https://www.aicrowd.com/challenges/citylearn-challenge-2023)
- **Framework:** [PyTorch](https://pytorch.org/)
- **LSTM:** Hochreiter & Schmidhuber (1997). Long Short-Term Memory. Neural Computation.
- **Attention:** Bahdanau et al. (2014). Neural Machine Translation by Jointly Learning to Align and Translate.
- **Transformer:** Vaswani et al. (2017). Attention is All You Need. NeurIPS.
