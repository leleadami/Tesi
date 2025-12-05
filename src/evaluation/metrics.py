"""
Metriche di valutazione per forecasting di serie temporali

Metriche:
- MAE, RMSE (metriche base)
- NMAE, NRMSE (metriche normalizzate)
- R² (coefficiente di determinazione)
- NSTD (deviazione standard normalizzata degli errori, per K-fold CV)
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Optional


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (Errore Assoluto Medio)"""
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (Radice dell'Errore Quadratico Medio)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Mean Absolute Error

    Normalizzato usando range (max - min) del ground truth

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        NMAE (0-1, lower is better)
    """
    mae_value = np.mean(np.abs(y_true - y_pred))
    y_range = np.max(y_true) - np.min(y_true)

    if y_range == 0:
        return 0.0

    return mae_value / y_range


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Root Mean Squared Error

    Normalizzato usando range (max - min) del ground truth

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        NRMSE (0-1, lower is better)
    """
    rmse_value = np.sqrt(np.mean((y_true - y_pred) ** 2))
    y_range = np.max(y_true) - np.min(y_true)

    if y_range == 0:
        return 0.0

    return rmse_value / y_range


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² Score (Coefficiente di Determinazione)"""
    return r2_score(y_true, y_pred)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcola tutte le metriche di valutazione

    Args:
        y_true: Valori ground truth (valori reali)
        y_pred: Valori predetti

    Returns:
        Dizionario con metriche: mae, rmse, nmae, nrmse, r2
    """
    metrics = {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'nmae': nmae(y_true, y_pred),
        'nrmse': nrmse(y_true, y_pred),
        'r2': r2(y_true, y_pred)
    }

    return metrics


def calculate_metrics_with_std(y_true_folds: list, y_pred_folds: list) -> Dict[str, float]:
    """
    Calculate metrics on concatenated K-fold arrays (CORRECT METHOD per professor)

    Procedure:
    1. Concatenate all k arrays of ground truth and predictions
    2. Calculate metrics (NMAE, NRMSE, etc.) ONCE on the total concatenated array
    3. Calculate NSTD (Normalized Standard Deviation of errors)

    Args:
        y_true_folds: List of ground truth arrays (one per fold)
        y_pred_folds: List of prediction arrays (one per fold)

    Returns:
        Dictionary with metrics: {metric_name: value}
        Includes 'nstd' = normalized standard deviation of errors
    """
    # Step 1: Concatenate all folds into single arrays
    y_true_all = np.concatenate(y_true_folds)
    y_pred_all = np.concatenate(y_pred_folds)

    # Step 2: Calculate metrics on concatenated array (NOT averaging per-fold metrics)
    metrics = calculate_metrics(y_true_all, y_pred_all)

    # Step 3: Calculate NSTD (Normalized Standard Deviation of errors)
    errors = y_true_all - y_pred_all
    y_range = np.max(y_true_all) - np.min(y_true_all)
    nstd = np.std(errors) / y_range if y_range > 0 else 0.0

    metrics['nstd'] = nstd

    return metrics


def denormalize_predictions(
    y_pred_norm: np.ndarray,
    scaler,
    feature_name: Optional[str] = None
) -> np.ndarray:
    """
    Denormalizza le predizioni alla scala originale

    Args:
        y_pred_norm: Predizioni normalizzate
        scaler: Scaler fittato (StandardScaler, MinMaxScaler, o dizionario di scaler)
        feature_name: Nome della feature (richiesto se scaler è un dizionario)

    Returns:
        Predizioni denormalizzate
    """
    if y_pred_norm.ndim == 1:
        y_pred_norm = y_pred_norm.reshape(-1, 1)

    # Se scaler è un dizionario, ottieni lo scaler specifico per la feature
    if isinstance(scaler, dict):
        if feature_name is None:
            raise ValueError("feature_name deve essere fornito quando scaler è un dizionario")
        scaler = scaler[feature_name]

    return scaler.inverse_transform(y_pred_norm).flatten()


def print_metrics(metrics: Dict[str, float], title: str = "Metriche di Valutazione"):
    """
    Stampa le metriche di valutazione in formato leggibile

    Args:
        metrics: Dizionario delle metriche
        title: Titolo per la stampa
    """
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)

    for metric_name, value in metrics.items():
        if metric_name == 'r2':
            print(f"{metric_name.upper():10s}: {value:8.4f}")
        else:
            print(f"{metric_name.upper():10s}: {value:8.4f}")

    print("=" * 60 + "\n")


def print_metrics_with_std(metrics: Dict[str, float], title: str = "Metriche K-Fold (Concatenate)"):
    """
    Stampa metriche K-fold calcolate su array concatenati

    Args:
        metrics: Dizionario {nome_metrica: valore}
                 Include NSTD (Deviazione Standard Normalizzata degli errori)
        title: Titolo per la stampa
    """
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)
    print("Metriche calcolate su TUTTE le predizioni dei fold concatenate:")
    print("-" * 70)

    # Stampa metriche in ordine
    metric_order = ['mae', 'rmse', 'nmae', 'nrmse', 'nstd', 'r2']

    for metric_name in metric_order:
        if metric_name in metrics:
            value = metrics[metric_name]
            if metric_name == 'nstd':
                print(f"{metric_name.upper():10s}: {value:8.4f}  (STD normalizzata degli errori)")
            else:
                print(f"{metric_name.upper():10s}: {value:8.4f}")

    print("=" * 70 + "\n")


# Retrocompatibilità: alias per nome vecchio della funzione
calculate_all_metrics = calculate_metrics
