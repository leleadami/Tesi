"""
Costanti globali del progetto

Questo file centralizza tutti i valori "magici" (magic numbers) usati nel progetto,
rendendo più facile modificarli globalmente e documentando il loro significato.
"""

from pathlib import Path

# ============================================================================
# PERCORSI DEL PROGETTO
# ============================================================================

# Percorso radice del progetto (risale 2 livelli da src/utils/constants.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Directory principali
DATA_ROOT = PROJECT_ROOT / 'data'
RESULTS_ROOT = PROJECT_ROOT / 'results'
CONFIGS_ROOT = PROJECT_ROOT / 'configs'

# ============================================================================
# CONFIGURAZIONE FASI DATASET
# ============================================================================

# Mapping fase → directory (lookup O(1), elimina catene if/elif)
PHASE_DIRS = {
    'phase_1': 'citylearn_challenge_2023_phase_1',
    'phase_2_local': 'citylearn_challenge_2023_phase_2_local',
    'phase_2_online': 'citylearn_challenge_2023_phase_2_online'
}

# Fasi valide per validazione input
VALID_PHASES = list(PHASE_DIRS.keys())

# ============================================================================
# CONFIGURAZIONE PLOT E VISUALIZZAZIONI
# ============================================================================

# Configurazione scatter plot
PLOT_CONFIG = {
    # Dimensione dei punti nei scatter plot
    'scatter_size': 40,

    # Trasparenza dei punti (0=trasparente, 1=opaco)
    'scatter_alpha': 0.95,

    # Spessore del bordo dei punti
    'edge_linewidth': 0.6,

    # Risoluzione immagini salvate (DPI = dots per inch)
    'figure_dpi': 300,

    # Trasparenza della griglia di sfondo
    'grid_alpha': 0.3,

    # Spessore linee griglia
    'grid_linewidth': 0.5,

    # Colore principale per scatter plot
    'scatter_color': '#1E5F8C',

    # Colore bordo punti
    'edge_color': 'darkblue',

    # Spessore linea ideale (y=x)
    'ideal_line_width': 2,

    # Trasparenza linea ideale
    'ideal_line_alpha': 0.7
}

# Configurazione residuals plot
RESIDUALS_CONFIG = {
    # Colore scatter residui
    'scatter_color': '#8B1A5A',

    # Colore bordo punti residui
    'edge_color': '#5A0F3A',

    # Dimensione punti residui
    'scatter_size': 40,

    # Trasparenza punti residui
    'scatter_alpha': 0.95
}

# ============================================================================
# CONFIGURAZIONE TRAINING
# ============================================================================

# Default training (sovrascritti da config YAML)
TRAINING_DEFAULTS = {
    'gradient_clip': 1.0,               # Norma max gradiente (anti-exploding)
    'early_stopping_patience': 15,      # Epoche senza miglioramento
    'lr_reduce_patience': 5,            # Epoche prima riduzione LR
    'lr_reduce_factor': 0.5,            # Fattore moltiplicativo LR
    'min_learning_rate': 1e-6,          # Soglia minima LR
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5                # Regolarizzazione L2
}

# ============================================================================
# CONFIGURAZIONE CROSS-VALIDATION
# ============================================================================

# Default sliding window cross-validation
CV_DEFAULTS = {
    'train_size': 0.4,      # Finestra training: 40%
    'test_size': 0.2,       # Finestra test: 20%
    'step': 0.1,            # Offset tra fold: 10%
    'n_folds_default': 5,   # Numero fold default
    'random_state': 42      # Seed riproducibilità
}

# ============================================================================
# CONFIGURAZIONE DATA PREPROCESSING
# ============================================================================

# Lista delle feature meteo che usano MinMaxScaler (valori limitati)
# Queste variabili hanno range noto e limitato (es. temperature, umidità)
MINMAX_FEATURES = [
    'outdoor_dry_bulb_temperature',
    'outdoor_relative_humidity',
    'diffuse_solar_irradiance',
    'direct_solar_irradiance',
    'month_sin',
    'month_cos',
    'hour_sin',
    'hour_cos',
    'day_type'
]

# Variabili target (carichi energetici) che usano StandardScaler (valori non limitati)
# Queste variabili possono assumere valori molto variabili e non limitati
# NOTA: heating_demand è sempre zero nel dataset estivo (giugno-agosto) quindi non è inclusa
TARGET_VARIABLES = [
    'cooling_demand',     # Domanda di raffreddamento (attiva in estate)
    'solar_generation',   # Generazione fotovoltaica
    'carbon_intensity'    # Intensità carbonica rete elettrica (kg CO2/kWh) - file separato
]

# Lunghezza sequenza input (lookback window)
# Quante ore guardare indietro per fare la previsione
SEQUENCE_LENGTH = 24

# Orizzonte di previsione (forecast horizon)
# Quante ore avanti prevedere
FORECAST_HORIZON = 1

# ============================================================================
# CONFIGURAZIONE DEVICE (CPU/GPU)
# ============================================================================

# Device di default per training
# Verrà sovrascritto automaticamente se GPU disponibile
DEFAULT_DEVICE = 'cpu'

# ============================================================================
# MESSAGGI DI ERRORE STANDARD
# ============================================================================

ERROR_MESSAGES = {
    'file_not_found': "File non trovato: {path}\nVerifica che il file esista nella directory corretta.",
    'invalid_phase': "Fase non valida: {phase}\nFasi valide: {valid_phases}",
    'invalid_building': "Building ID non valido: {building_id}\nBuilding validi: 1, 2, 3",
    'invalid_target': "Target non valido: {target}\nTarget validi: {valid_targets}",
    'csv_parse_error': "Errore nel parsing del file CSV: {path}\nErrore: {error}",
    'gpu_not_available': "GPU non disponibile, uso CPU. Per usare GPU installa CUDA e PyTorch con supporto CUDA.",
    'checkpoint_not_found': "Checkpoint non trovato: {path}\nVerifica che il modello sia stato trainato."
}
