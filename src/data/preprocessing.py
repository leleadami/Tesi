"""
Pipeline di preprocessing dei dati per CityLearn Challenge 2023

Gestisce:
- Caricamento dati building e meteo
- Merge dei dataset
- Gestione valori mancanti (forward fill, interpolazione, ecc.)
- Feature engineering (features temporali)
- Normalizzazione (MinMaxScaler per variabili limitate, StandardScaler per carichi)
- Creazione sequenze sliding window per forecasting time series
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
from typing import Tuple, List, Dict, Any

# Importa costanti e gestione errori centralizzate
from utils.constants import PHASE_DIRS, VALID_PHASES, DATA_ROOT, TARGET_VARIABLES
from utils.error_handling import (
    raise_file_not_found,
    raise_invalid_phase,
    raise_invalid_building,
    raise_csv_parse_error,
    raise_missing_columns,
    DataLoadError
)


def load_building_data(building_id: int, phase: str = "phase_1", data_dir: Path | str | None = None) -> pd.DataFrame:
    """
    Carica i dati di un building specifico per una determinata fase del dataset.

    Questa funzione gestisce il caricamento dei file CSV contenenti i dati di consumo
    energetico di un singolo edificio (Building_1, Building_2, o Building_3).

    Args:
        building_id: ID dell'edificio (1, 2, o 3)
        phase: Fase del dataset ("phase_1", "phase_2_local", "phase_2_online")
        data_dir: Directory base dei dati (se None, usa percorso default relativo)

    Returns:
        DataFrame contenente i dati dell'edificio con timestamp parsato

    Raises:
        InvalidBuildingError: Se building_id non è 1, 2, o 3
        InvalidPhaseError: Se phase non è una fase valida
        DataLoadError: Se il file CSV non viene trovato o è corrotto
    """
    # Validazione building_id (deve essere 1, 2 o 3)
    if building_id not in [1, 2, 3]:
        raise_invalid_building(building_id)

    # Validazione fase del dataset
    if phase not in VALID_PHASES:
        raise_invalid_phase(phase, VALID_PHASES)

    # Determinazione directory dati: usa percorso fornito o default
    if data_dir is None:
        data_dir = DATA_ROOT
    else:
        data_dir = Path(data_dir)

    # Costruzione percorso tramite lookup O(1) invece di catena if/elif
    phase_dir = PHASE_DIRS[phase]
    path = data_dir / phase_dir / f"Building_{building_id}.csv"

    # Caricamento CSV con gestione centralizzata degli errori
    df: pd.DataFrame | None = None
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        # Lista file disponibili per facilitare il debug
        try:
            available_files = [f.name for f in (data_dir / phase_dir).glob("Building_*.csv")]
        except:
            available_files = None
        raise_file_not_found(path, "CSV dati building", available_files)
    except pd.errors.ParserError as e:
        raise_csv_parse_error(path, e)
    except Exception as e:
        raise DataLoadError(
            f"Errore imprevisto nel caricamento di {path}",
            f"Dettagli: {str(e)}"
        )

    # Garanzia per il type checker: df è definito se l'esecuzione raggiunge questo punto
    assert df is not None

    # Conversione timestamp da stringa a datetime per operazioni temporali
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']) 

    return df


def load_weather_data(phase: str = "phase_1", data_dir: Path | str | None = None) -> pd.DataFrame:
    """
    Carica i dati meteorologici per una determinata fase del dataset.

    I dati meteo sono condivisi tra tutti i building (non sono specifici per edificio)
    e contengono variabili come temperatura, radiazione solare, umidità, ecc.

    Args:
        phase: Fase del dataset ("phase_1", "phase_2_local", "phase_2_online")
        data_dir: Directory base dei dati (se None, usa percorso default relativo)

    Returns:
        DataFrame contenente i dati meteorologici con timestamp parsato

    Raises:
        InvalidPhaseError: Se phase non è una fase valida
        DataLoadError: Se il file CSV non viene trovato o è corrotto
    """
    # Validazione fase del dataset
    if phase not in VALID_PHASES:
        raise_invalid_phase(phase, VALID_PHASES)

    # Determinazione directory dati: usa percorso fornito o default
    if data_dir is None:
        data_dir = DATA_ROOT
    else:
        data_dir = Path(data_dir)

    # Costruzione percorso tramite lookup O(1)
    phase_dir = PHASE_DIRS[phase]
    path = data_dir / phase_dir / "weather.csv"

    # Caricamento CSV con gestione centralizzata degli errori
    df: pd.DataFrame | None = None
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise_file_not_found(path, "CSV dati meteorologici")
    except pd.errors.ParserError as e:
        raise_csv_parse_error(path, e)
    except Exception as e:
        raise DataLoadError(
            f"Errore imprevisto nel caricamento di {path}",
            f"Dettagli: {str(e)}"
        )

    # Garanzia per il type checker: df è definito se l'esecuzione raggiunge questo punto
    assert df is not None

    # Conversione timestamp per merge con dati building
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']) 

    return df


def load_carbon_intensity(phase: str = "phase_1", data_dir: Path | str | None = None) -> pd.DataFrame:
    """
    Carica i dati di carbon intensity per una determinata fase.

    Carbon intensity rappresenta l'intensità carbonica della rete elettrica
    (kg CO2 emessi per kWh consumato), dipende dal mix energetico.

    Args:
        phase: Fase del dataset ("phase_1", "phase_2_local", "phase_2_online")
        data_dir: Directory base dei dati (se None, usa percorso default)

    Returns:
        DataFrame con colonna 'carbon_intensity' (kg CO2/kWh)

    Raises:
        InvalidPhaseError: Se phase non è valida
        DataLoadError: Se il file non viene trovato o è corrotto
    """
    # Validazione fase del dataset
    if phase not in VALID_PHASES:
        raise_invalid_phase(phase, VALID_PHASES)

    # Determinazione directory dati
    if data_dir is None:
        data_dir = DATA_ROOT
    else:
        data_dir = Path(data_dir)

    # Costruzione percorso file
    phase_dir = PHASE_DIRS[phase]
    path = data_dir / phase_dir / "carbon_intensity.csv"

    # Caricamento CSV con gestione centralizzata degli errori
    df: pd.DataFrame | None = None
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        try:
            available_files = [f.name for f in (data_dir / phase_dir).glob("*.csv")]
        except:
            available_files = None
        raise_file_not_found(path, "CSV carbon intensity", available_files)
    except pd.errors.ParserError as e:
        raise_csv_parse_error(path, e)
    except Exception as e:
        raise DataLoadError(
            f"Errore imprevisto nel caricamento di {path}",
            f"Dettagli: {str(e)}"
        )

    # Garanzia per il type checker: df è definito se l'esecuzione raggiunge questo punto
    assert df is not None

    return df


def merge_building_weather(building_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unisce i dati di un edificio con i dati meteorologici.

    Questa funzione combina le informazioni specifiche del building (consumi energetici)
    con le variabili meteo (temperatura, radiazione solare, ecc.) che sono condivise.

    Strategia di merge:
    - Se entrambi i DataFrame hanno colonna 'timestamp' → merge su timestamp (preferito)
    - Altrimenti → concatenazione per colonne assumendo allineamento temporale

    Args:
        building_df: DataFrame con dati specifici del building
        weather_df: DataFrame con dati meteorologici (condivisi tra building)

    Returns:
        DataFrame unito contenente sia dati building che meteo
    """
    # Strategia di merge basata sulla presenza del timestamp
    if 'timestamp' in building_df.columns and 'timestamp' in weather_df.columns:
        # Merge su timestamp: garantisce allineamento corretto indipendentemente dall'ordine
        merged = building_df.merge(weather_df, on='timestamp', how='left')
    else:
        # Fallback: concatenazione assumendo allineamento temporale
        merged = pd.concat([building_df, weather_df], axis=1)

    return merged


def remove_useless_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove feature inutili identificate tramite analisi.

    Feature rimosse:
    1. heating_demand - Sempre 0.0 nel dataset estivo (giugno-agosto)
    2. daylight_savings_status - Costante (sempre 0)
    3. hvac_mode - Costante (sempre 1)
    4. indoor_dry_bulb_temperature_cooling_set_point - Ridondante (≈ indoor_dry_bulb_temperature)
    5. indoor_dry_bulb_temperature_heating_set_point - Ridondante (≈ indoor_dry_bulb_temperature)

    Rationale:
    - Feature costanti hanno varianza 0 → nessuna informazione utile
    - Feature ridondanti aumentano dimensionalità senza aggiungere segnale
    - Rimozione riduce overfitting e migliora generalizzazione

    Args:
        df: DataFrame dopo merge building + weather

    Returns:
        DataFrame con feature inutili rimosse
    """
    # Feature da rimuovere (costanti o ridondanti)
    useless_features = [
        'heating_demand',                                  # Sempre 0 (dataset estivo)
        'daylight_savings_status',                         # Costante: 0
        'hvac_mode',                                       # Costante: 1
        'indoor_dry_bulb_temperature_cooling_set_point',   # Ridondante
        'indoor_dry_bulb_temperature_heating_set_point'    # Ridondante
    ]

    # Rimozione colonne (errors='ignore' previene eccezioni per colonne già assenti)
    df_cleaned = df.drop(columns=useless_features, errors='ignore')

    # Log informativo (opzionale)
    removed_count = len([col for col in useless_features if col in df.columns])
    if removed_count > 0:
        print(f"[Preprocessing] Rimosse {removed_count} feature inutili")

    return df_cleaned


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge features temporali con encoding ciclico (sin/cos).

    RAZIONALE ENCODING CICLICO:
    Variabili temporali (ora, mese) sono intrinsecamente cicliche ma numericamente discontinue:
    - 23:00 ↔ 00:00: vicine temporalmente, distanti numericamente (23 vs 0)
    - Dicembre ↔ Gennaio: adiacenti ma lontani (12 vs 1)

    Encoding sin/cos preserva la continuità ciclica mappando valori su cerchio unitario:
    - sin(2π·x/period) e cos(2π·x/period) formano coordinate cartesiane sul cerchio
    - Distanze euclidee nello spazio (sin, cos) riflettono prossimità temporale

    FORMULA:
    - hour_sin/cos = sin/cos(2π · hour / 24)   → proiezione 24-ore su cerchio
    - month_sin/cos = sin/cos(2π · (month-1) / 12) → proiezione 12-mesi su cerchio

    Args:
        df: DataFrame contenente colonne 'hour', 'month', 'day_type'

    Returns:
        DataFrame con features temporali aggiunte (hour_sin, hour_cos, month_sin, month_cos)
    """
    # Copia per evitare side effects sul DataFrame originale
    df = df.copy()

    # Encoding ciclico ora (0-23)
    if 'hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Encoding ciclico mese (1-12, shift a 0-11 per simmetria)
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

    # day_type rimane categorico (0-6), encoding ciclico non necessario

    return df


def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing values in the dataset

    Args:
        df: DataFrame with potential missing values
        method: Imputation method
            - 'ffill': Forward fill (use last valid observation)
            - 'bfill': Backward fill (use next valid observation)
            - 'interpolate': Linear interpolation
            - 'drop': Drop rows with missing values
            - 'mean': Fill with column mean

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    # Check if there are any missing values
    missing_count = df.isnull().sum().sum()

    if missing_count == 0:
        return df

    print(f"WARNING: Found {missing_count} missing values. Applying '{method}' imputation...")

    if method == 'ffill':
        # Forward fill: use last valid observation
        df = df.ffill()
        # If first rows are NaN, use backward fill
        df = df.bfill()

    elif method == 'bfill':
        # Backward fill: use next valid observation
        df = df.bfill()
        # If last rows are NaN, use forward fill
        df = df.ffill()

    elif method == 'interpolate':
        # Linear interpolation for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

    elif method == 'mean':
        # Fill with column mean (only for numeric)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    elif method == 'drop':
        # Drop rows with any missing values
        original_len = len(df)
        df = df.dropna()
        dropped = original_len - len(df)
        print(f"  Dropped {dropped} rows ({dropped/original_len*100:.2f}%)")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'ffill', 'bfill', 'interpolate', 'mean', or 'drop'")

    # Verify no missing values remain
    remaining = df.isnull().sum().sum()
    if remaining > 0:
        print(f"WARNING: {remaining} missing values still remain after imputation!")
    else:
        print(f"All missing values handled successfully")

    return df



# Strategia di normalizzazione differenziata:
# - Variabili limitate (temperatura, umidità) → MinMaxScaler [0,1]
# - Variabili di carico/energia → StandardScaler (μ=0, σ=1)
# - Feature temporali sin/cos → nessuna scalatura (già in [-1,1])

def normalize_features(
    df: pd.DataFrame,
    scalers: Dict[str, Any] | None = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize features using MinMaxScaler for bounded vars, StandardScaler for loads

    Args:
        df: DataFrame with features
        scalers: Dictionary of fitted scalers (if fit=False)
        fit: Whether to fit scalers or use existing ones

    Returns:
        Tuple of (normalized DataFrame, scalers dict)
    """
    df_norm = df.copy()

    if scalers is None:
        scalers = {}

    # Feature limitate: MinMaxScaler [0,1]
    bounded_features = [
        'outdoor_dry_bulb_temperature',
        'outdoor_relative_humidity',
        'diffuse_solar_irradiance',
        'direct_solar_irradiance',
        'indoor_dry_bulb_temperature'
    ]

    # Feature di carico: StandardScaler (μ=0, σ=1)
    load_features = [
        'cooling_demand',
        'heating_demand',
        'non_shiftable_load',
        'solar_generation',
        'carbon_intensity'
    ]

    # Feature temporali sin/cos già in [-1,1]: nessuna normalizzazione

    # Applica MinMaxScaler a feature limitate
    for feature in bounded_features:
        if feature in df_norm.columns:
            if fit:
                scaler = MinMaxScaler()
                df_norm[feature] = scaler.fit_transform(df_norm[[feature]])
                scalers[feature] = scaler
            else:
                df_norm[feature] = scalers[feature].transform(df_norm[[feature]])

    # Applica StandardScaler a feature di carico
    for feature in load_features:
        if feature in df_norm.columns:
            if fit:
                scaler = StandardScaler()
                df_norm[feature] = scaler.fit_transform(df_norm[[feature]])
                scalers[feature] = scaler
            else:
                df_norm[feature] = scalers[feature].transform(df_norm[[feature]])

    return df_norm, scalers



# Sliding window per serie temporali:
# X[i] = feature[i : i+sequence_length]               → finestra lookback
# y[i] = target[i + sequence_length + forecast_horizon - 1]  → valore futuro

# Esempio (sequence_length=3, forecast_horizon=1):
# Time:   t0  t1  t2  t3  t4  t5
# Data:    1   2   3   4   5   6
# Target: 10  20  30  40  50  60
# ────────────────────────────────
# X[0] = [1,2,3]  → y[0] = 40  (predice t3)
# X[1] = [2,3,4]  → y[1] = 50  (predice t4)
# X[2] = [3,4,5]  → y[2] = 60  (predice t5)


def create_sequences(
    data: np.ndarray | Any,
    target: np.ndarray | Any,
    sequence_length: int = 24,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea sequenze sliding window per forecasting time series.

    Args:
        data: Array feature di input (T, F) - accetta np.ndarray o pandas .values
        target: Array target (T,) - accetta np.ndarray o pandas .values
        sequence_length: Finestra lookback (ore)
        forecast_horizon: Orizzonte di predizione (ore avanti)

    Returns:
        Tupla (X, y):
        - X: (N, sequence_length, F) - sequenze di input
        - y: (N,) - valori target
    """
    # Conversione a numpy per compatibilità universale (pandas, list, ecc.)
    data = np.asarray(data)
    target = np.asarray(target)

    X, y = [], []

    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:i+sequence_length])
        y.append(target[i + sequence_length + forecast_horizon - 1])

    return np.array(X), np.array(y)



# Pipeline preprocessing con split pre-normalizzazione (prevenzione data leakage):
# 1. Carica dati building + meteo
# 2. Merge su timestamp
# 3. Gestione valori mancanti
# 4. Split train/val (CRITICO: prima della normalizzazione)
# 5. Fit scalers solo su train, applica a val
# 6. Feature engineering temporali
# 7. Generazione sequenze sliding window

def prepare_train_val_split(
    building_id: int,
    phase: str,
    target_variable: str = "cooling_demand",
    weather_features: List[str] | None = None,
    sequence_length: int = 24,
    forecast_horizon: int = 1,
    val_split: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], List[str]]:
    """
    Prepare dataset with train/val split BEFORE normalization (best practice, no data leakage)

    Pipeline:
    1. Load and merge data
    2. Handle missing values
    3. Add temporal features
    4. Split train/val (80/20)
    5. Fit scalers ONLY on train
    6. Apply scalers to val
    7. Create sequences

    Args:
        building_id: 1, 2, or 3
        phase: "phase_1", "phase_2_local", etc.
        target_variable: Target to forecast
        weather_features: Weather features to include
        sequence_length: Lookback window
        forecast_horizon: Prediction horizon
        val_split: Fraction for validation (default 0.2 = 20%)

    Returns:
        Tuple of (X_train, X_val, y_train, y_val, scalers, feature_names)
    """
    if weather_features is None:
        weather_features = [
            'outdoor_dry_bulb_temperature',
            'outdoor_relative_humidity',
            'diffuse_solar_irradiance',
            'direct_solar_irradiance'
        ]

    # Caricamento e preprocessing preliminare
    building_df = load_building_data(building_id, phase)
    weather_df = load_weather_data(phase)
    df = merge_building_weather(building_df, weather_df)
    df = handle_missing_values(df, method='ffill')
    df = add_temporal_features(df)

    # Selezione feature
    feature_columns = weather_features + ['hour_sin', 'hour_cos']
    if 'indoor_dry_bulb_temperature' in df.columns:
        feature_columns.append('indoor_dry_bulb_temperature')
    feature_columns.append(target_variable)
    feature_columns = [f for f in feature_columns if f in df.columns]

    df_subset = df[feature_columns + ['month', 'day_type']].copy()

    # Split temporale PRIMA della normalizzazione (anti-leakage)
    split_idx = int(len(df_subset) * (1 - val_split))
    df_train = df_subset.iloc[:split_idx].copy()
    df_val = df_subset.iloc[split_idx:].copy()

    # Normalizzazione: fit solo su train, transform su entrambi
    df_train_norm, scalers = normalize_features(df_train, scalers=None, fit=True)
    df_val_norm, _ = normalize_features(df_val, scalers=scalers, fit=False)

    # Generazione sequenze sliding window
    X_train_features = df_train_norm[feature_columns].values
    y_train_target = df_train_norm[target_variable].values
    X_val_features = df_val_norm[feature_columns].values
    y_val_target = df_val_norm[target_variable].values

    X_train, y_train = create_sequences(X_train_features, y_train_target, sequence_length, forecast_horizon)
    X_val, y_val = create_sequences(X_val_features, y_val_target, sequence_length, forecast_horizon)

    return X_train, X_val, y_train, y_val, scalers, feature_columns


def prepare_dataset(
    building_id: int,
    phase: str,
    target_variable: str = "cooling_demand",
    weather_features: List[str] | None = None,
    sequence_length: int = 24,
    forecast_horizon: int = 1,
    scalers: Dict[str, Any] | None = None,
    fit_scalers: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[str]]:
    """
    Complete preprocessing pipeline: load → merge → engineer → normalize → sequences

    Args:
        building_id: 1, 2, or 3
        phase: "phase_1", "phase_2_local", etc.
        target_variable: Target to forecast
        weather_features: Weather features to include
        sequence_length: Lookback window
        forecast_horizon: Prediction horizon
        scalers: Pre-fitted scalers (if fit_scalers=False)
        fit_scalers: Whether to fit scalers

    Returns:
        Tuple of (X, y, scalers, feature_names)
        - X: (N, sequence_length, F)
        - y: (N,)
        - scalers: Dictionary of fitted scalers
        - feature_names: List of feature names
    """
    if weather_features is None:
        weather_features = [
            'outdoor_dry_bulb_temperature',
            'outdoor_relative_humidity',
            'diffuse_solar_irradiance',
            'direct_solar_irradiance'
        ]

    # 1. Load data
    building_df = load_building_data(building_id, phase)
    weather_df = load_weather_data(phase)

    # 2. Merge
    df = merge_building_weather(building_df, weather_df)

    # 3. Remove useless features (constant/redundant)
    df = remove_useless_features(df)

    # 4. Handle missing values (if any)
    df = handle_missing_values(df, method='ffill')

    # 5. Add temporal features
    df = add_temporal_features(df)

    # 5. Select features
    feature_columns = weather_features + ['hour_sin', 'hour_cos']

    # Add building features
    if 'indoor_dry_bulb_temperature' in df.columns:
        feature_columns.append('indoor_dry_bulb_temperature')

    # Add target as autoregressive feature
    feature_columns.append(target_variable)

    # Ensure all features exist
    feature_columns = [f for f in feature_columns if f in df.columns]

    # 6. Normalize
    df_subset = df[feature_columns + ['month', 'day_type']].copy()
    df_norm, scalers = normalize_features(df_subset, scalers=scalers, fit=fit_scalers)

    # 7. Prepare arrays
    X_features = df_norm[feature_columns].values
    y_target = df_norm[target_variable].values

    # 8. Create sequences
    X, y = create_sequences(X_features, y_target, sequence_length, forecast_horizon)

    return X, y, scalers, feature_columns



def prepare_carbon_intensity_dataset(
    phase: str,
    weather_features: List[str] | None = None,
    sequence_length: int = 24,
    forecast_horizon: int = 1,
    scalers: Dict[str, Any] | None = None,
    fit_scalers: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[str]]:
    """
    Prepara dataset per forecasting di carbon_intensity.

    Carbon intensity è una misura aggregata a livello di rete elettrica,
    non specifica per building. Viene predetta usando feature meteo e temporali.

    Args:
        phase: Fase del dataset ("phase_1", "phase_2_local", "phase_2_online")
        weather_features: Feature meteo da includere (default: temp, umidità, irradianze)
        sequence_length: Finestra lookback (default: 24 ore)
        forecast_horizon: Orizzonte predizione (default: 1 ora)
        scalers: Scalers pre-fitted (se fit_scalers=False)
        fit_scalers: Se True, fit scalers; se False, usa scalers forniti

    Returns:
        Tuple di (X, y, scalers, feature_names)
        - X: (N, sequence_length, num_features)
        - y: (N,)
        - scalers: Dictionary di scalers fitted
        - feature_names: Lista nomi feature

    Raises:
        InvalidPhaseError: Se phase non valida
        DataLoadError: Se file non trovato/corrotto
    """
    if weather_features is None:
        weather_features = [
            'outdoor_dry_bulb_temperature',
            'outdoor_relative_humidity',
            'diffuse_solar_irradiance',
            'direct_solar_irradiance'
        ]

    # 1. Carica dati
    carbon_df = load_carbon_intensity(phase)
    weather_df = load_weather_data(phase)

    # 2. Merge per colonne (assumendo allineamento temporale)
    df = pd.concat([carbon_df, weather_df], axis=1)

    # 3. Rimuovi feature inutili (se presenti)
    df = remove_useless_features(df)

    # 4. Gestisci valori mancanti
    df = handle_missing_values(df, method='ffill')

    # 5. Aggiungi feature temporali
    df = add_temporal_features(df)

    # 6. Seleziona feature
    target_variable = 'carbon_intensity'
    feature_columns = weather_features + ['hour_sin', 'hour_cos', target_variable]
    feature_columns = [f for f in feature_columns if f in df.columns]

    # 7. Normalizzazione
    df_norm, scalers = normalize_features(df[feature_columns], scalers=scalers, fit=fit_scalers)

    # 8. Prepara array
    X_features = df_norm[feature_columns].values
    y_target = df_norm[target_variable].values

    # 9. Crea sequenze
    X, y = create_sequences(X_features, y_target, sequence_length, forecast_horizon)

    return X, y, scalers, feature_columns


def prepare_neighborhood_train_val_split(
    building_ids: List[int],
    phase: str,
    target_variable: str = "cooling_demand",
    weather_features: List[str] | None = None,
    sequence_length: int = 24,
    forecast_horizon: int = 1,
    val_split: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], List[str]]:
    """
    Prepare aggregated neighborhood dataset with train/val split BEFORE normalization.

    This function prevents data leakage by:
    1. Aggregating building data (summing target variables)
    2. Splitting into train/val BEFORE normalization
    3. Fitting scalers ONLY on train
    4. Applying same scalers to val
    5. Creating sequences for both

    Args:
        building_ids: List of building IDs to aggregate (e.g., [1, 2, 3])
        phase: Data phase
        target_variable: Target to forecast (cooling_demand, solar_generation, carbon_intensity)
        weather_features: Weather features to include
        sequence_length: Lookback window
        forecast_horizon: Prediction horizon
        val_split: Fraction for validation (default 0.2 = 20%)

    Returns:
        Tuple of (X_train, X_val, y_train, y_val, scalers, feature_names)

    Note:
        For carbon_intensity (network-level), use prepare_carbon_intensity_dataset() instead,
        as it doesn't require building aggregation.
    """
    if weather_features is None:
        weather_features = [
            'outdoor_dry_bulb_temperature',
            'outdoor_relative_humidity',
            'diffuse_solar_irradiance',
            'direct_solar_irradiance'
        ]

    # 1. Load weather once (shared across buildings)
    weather_df = load_weather_data(phase)

    # 2. Aggregate building target variable (sum across buildings)
    aggregated_target = None

    for building_id in building_ids:
        building_df = load_building_data(building_id, phase)

        if aggregated_target is None:
            aggregated_target = building_df[[target_variable]].copy()
        else:
            # Sum target values across buildings
            aggregated_target += building_df[[target_variable]].values

    # 3. Merge aggregated target with weather
    df = pd.concat([aggregated_target, weather_df], axis=1)

    # 4. Remove useless features
    df = remove_useless_features(df)

    # 5. Handle missing values
    df = handle_missing_values(df, method='ffill')

    # 6. Add temporal features
    df = add_temporal_features(df)

    # 7. Select features
    feature_columns = weather_features + ['hour_sin', 'hour_cos', target_variable]
    feature_columns = [f for f in feature_columns if f in df.columns]

    df_subset = df[feature_columns + ['month', 'day_type']].copy()

    # 8. SPLIT BEFORE NORMALIZATION (prevents data leakage!)
    split_idx = int(len(df_subset) * (1 - val_split))
    df_train = df_subset.iloc[:split_idx].copy()
    df_val = df_subset.iloc[split_idx:].copy()

    # 9. Normalize: fit on train only, apply to val
    df_train_norm, scalers = normalize_features(df_train, scalers=None, fit=True)
    df_val_norm, _ = normalize_features(df_val, scalers=scalers, fit=False)

    # 10. Create sequences
    X_train_features = df_train_norm[feature_columns].values
    y_train_target = df_train_norm[target_variable].values
    X_val_features = df_val_norm[feature_columns].values
    y_val_target = df_val_norm[target_variable].values

    X_train, y_train = create_sequences(X_train_features, y_train_target, sequence_length, forecast_horizon)
    X_val, y_val = create_sequences(X_val_features, y_val_target, sequence_length, forecast_horizon)

    return X_train, X_val, y_train, y_val, scalers, feature_columns
