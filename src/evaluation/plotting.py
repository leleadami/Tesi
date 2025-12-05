"""
Funzioni di plotting per valutazione e visualizzazione risultati

Questo modulo fornisce funzioni per creare grafici professionali dei risultati:
- Scatter plot predizioni vs ground truth
- Grafici con residui per analisi errori
- Comparazione fold in cross-validation
- Training history multi-fold

Convenzione grafici:
- Asse X: Ground truth (valori reali)
- Asse Y: Prediction (valori predetti)
- Titolo: Metriche (NMAE, NRMSE, NSTD quando applicabile)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
from .metrics import nmae, nrmse

# Importa configurazioni grafiche centralizzate
from utils.constants import PLOT_CONFIG, RESIDUALS_CONFIG


def plot_scatter_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction vs Ground Truth",
    save_path: Optional[Path] = None,
    show_metrics: bool = True,
    figsize: tuple = (10, 10),
    alpha: float = 0.6,
    color: str = '#2E86AB',
    nmae_value: Optional[float] = None,
    nrmse_value: Optional[float] = None,
    nstd_value: Optional[float] = None
):
    """
    Crea scatter plot ground truth vs prediction

    Args:
        y_true: Ground truth values (asse x)
        y_pred: Predicted values (asse y)
        title: Base title (metrics saranno aggiunte se show_metrics=True)
        save_path: Path dove salvare il plot (opzionale)
        show_metrics: Se True, aggiunge NMAE/NRMSE/NSTD al titolo
        figsize: Dimensione figura
        alpha: Trasparenza punti
        color: Colore punti
        nmae_value: NMAE pre-calcolato (opzionale)
        nrmse_value: NRMSE pre-calcolato (opzionale)
        nstd_value: NSTD pre-calcolato (opzionale, for K-fold)
    """
    # Calcola metriche se non fornite
    if show_metrics:
        if nmae_value is None:
            nmae_value = nmae(y_true, y_pred)
        if nrmse_value is None:
            nrmse_value = nrmse(y_true, y_pred)

    # Crea figura con stile professionale
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot con configurazione centralizzata da constants.py
    # Usa valori definiti in PLOT_CONFIG per consistency tra tutti i grafici
    ax.scatter(
        y_true, y_pred,
        alpha=PLOT_CONFIG['scatter_alpha'],      # Trasparenza punti (0.95)
        s=PLOT_CONFIG['scatter_size'],           # Dimensione punti (40)
        color=PLOT_CONFIG['scatter_color'],      # Colore principale (#1E5F8C)
        edgecolors=PLOT_CONFIG['edge_color'],    # Colore bordo (darkblue)
        linewidth=PLOT_CONFIG['edge_linewidth']  # Spessore bordo (0.6)
    )

    # Calcola range assi: min deve essere >= 0 per dati energetici (sempre positivi)
    # max è il valore massimo tra ground truth e predizioni
    min_val = max(0, min(y_true.min(), y_pred.min()))
    max_val = max(y_true.max(), y_pred.max())

    # Linea ideale (y = x): dove predizione = ground truth (predizione perfetta)
    # Ogni punto sulla linea rappresenta un'osservazione predetta perfettamente
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        'r--',  # Rosso tratteggiato
        lw=PLOT_CONFIG['ideal_line_width'],      # Spessore linea (2)
        label='Perfect Prediction',
        alpha=PLOT_CONFIG['ideal_line_alpha']    # Trasparenza (0.7)
    )

    # Imposta limiti assi espliciti per garantire aspect ratio 1:1
    # Questo evita che matplotlib auto-scali e distorca il grafico
    ax.set_xlim((min_val, max_val))
    ax.set_ylim((min_val, max_val))

    # Griglia di sfondo per facilitare lettura valori
    ax.grid(
        True,
        alpha=PLOT_CONFIG['grid_alpha'],         # Trasparenza griglia (0.3)
        linestyle='--',
        linewidth=PLOT_CONFIG['grid_linewidth']  # Spessore linee griglia (0.5)
    )

    # Labels con font pesante
    ax.set_xlabel('Ground Truth', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prediction', fontsize=14, fontweight='bold')

    # Titolo con metriche
    if show_metrics:
        if nstd_value is not None:
            # K-fold: mostra NMAE, NRMSE, NSTD
            metrics_str = f'NMAE: {nmae_value:.4f} | NRMSE: {nrmse_value:.4f} | NSTD: {nstd_value:.4f}'
        else:
            # Single run: solo NMAE e NRMSE
            metrics_str = f'NMAE: {nmae_value:.4f} | NRMSE: {nrmse_value:.4f}'

        full_title = f'{title}\n{metrics_str}'
    else:
        full_title = title

    ax.set_title(full_title, fontsize=16, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Aspect ratio uguale per mantenere proporzioni corrette
    ax.set_aspect('equal', adjustable='box')

    # Layout tight
    plt.tight_layout()

    # Salva grafico su file se richiesto
    if save_path:
        # DPI centralizzata da constants per qualità consistente
        plt.savefig(save_path, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        print(f"✓ Scatter plot salvato: {save_path}")

    return fig, ax


def plot_scatter_with_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction Analysis",
    save_path: Optional[Path] = None,
    figsize: tuple = (16, 6),
    nmae_value: Optional[float] = None,
    nrmse_value: Optional[float] = None,
    nmae_std: Optional[float] = None,
    nrmse_std: Optional[float] = None
):
    """
    Crea figura con 2 subplot: scatter plot + residuals

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Base title
        save_path: Path dove salvare
        figsize: Dimensione figura
        nmae_value, nrmse_value: Metriche pre-calcolate (opzionali)
        nmae_std, nrmse_std: Std delle metriche (opzionali, per K-fold)
    """
    # Calcola metriche se non fornite
    if nmae_value is None:
        nmae_value = nmae(y_true, y_pred)
    if nrmse_value is None:
        nrmse_value = nrmse(y_true, y_pred)

    # Calcola residuals
    residuals = y_pred - y_true

    # Crea figura
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- SUBPLOT 1: Scatter plot ---
    ax1 = axes[0]

    # Scatter plot predizioni con configurazione centralizzata
    # Usa gli stessi parametri di plot_scatter_predictions() per consistency
    ax1.scatter(
        y_true, y_pred,
        alpha=PLOT_CONFIG['scatter_alpha'],      # Trasparenza punti (0.95)
        s=PLOT_CONFIG['scatter_size'],           # Dimensione punti (40)
        color=PLOT_CONFIG['scatter_color'],      # Colore principale (#1E5F8C)
        edgecolors=PLOT_CONFIG['edge_color'],    # Colore bordo (darkblue)
        linewidth=PLOT_CONFIG['edge_linewidth']  # Spessore bordo (0.6)
    )

    # Calcola range assi: min >= 0 per dati energetici (sempre positivi)
    min_val = max(0, min(y_true.min(), y_pred.min()))
    max_val = max(y_true.max(), y_pred.max())

    # Linea ideale (y = x): predizione perfetta
    ax1.plot(
        [min_val, max_val], [min_val, max_val],
        'r--',
        lw=PLOT_CONFIG['ideal_line_width'],      # Spessore linea (2)
        label='Perfect Prediction',
        alpha=PLOT_CONFIG['ideal_line_alpha']    # Trasparenza (0.7)
    )

    # Imposta limiti assi espliciti per garantire aspect ratio 1:1
    ax1.set_xlim((min_val, max_val))
    ax1.set_ylim((min_val, max_val))

    # Griglia di sfondo
    ax1.grid(
        True,
        alpha=PLOT_CONFIG['grid_alpha'],         # Trasparenza griglia (0.3)
        linestyle='--',
        linewidth=PLOT_CONFIG['grid_linewidth']  # Spessore linee (0.5)
    )

    ax1.set_xlabel('Ground Truth', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Prediction', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction vs Ground Truth', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Aspect ratio uguale per mantenere proporzioni corrette
    ax1.set_aspect('equal', adjustable='box')

    # --- SUBPLOT 2: Residuals (Analisi degli errori) ---
    ax2 = axes[1]

    # Scatter plot residui con configurazione centralizzata da RESIDUALS_CONFIG
    # Residui = Pred - True: positivo = sovrastima, negativo = sottostima
    # Analisi visiva: pattern nei residui indicano bias del modello
    ax2.scatter(
        y_true, residuals,
        alpha=RESIDUALS_CONFIG['scatter_alpha'],      # Trasparenza punti (0.95)
        s=RESIDUALS_CONFIG['scatter_size'],           # Dimensione punti (40)
        color=RESIDUALS_CONFIG['scatter_color'],      # Colore residui (#8B1A5A)
        edgecolors=RESIDUALS_CONFIG['edge_color'],    # Colore bordo (#5A0F3A)
        linewidth=PLOT_CONFIG['edge_linewidth']       # Spessore bordo (0.6)
    )

    # Linea orizzontale a y=0: indica errore nullo (predizione perfetta)
    # Punti sopra la linea = sovrastima, punti sotto = sottostima
    ax2.axhline(
        y=0,
        color='r',
        linestyle='--',
        lw=PLOT_CONFIG['ideal_line_width'],  # Spessore linea (2)
        alpha=PLOT_CONFIG['ideal_line_alpha'],  # Trasparenza (0.7)
        label='Zero Error'
    )

    # Griglia di sfondo per facilitare lettura residui
    ax2.grid(
        True,
        alpha=PLOT_CONFIG['grid_alpha'],         # Trasparenza griglia (0.3)
        linestyle='--',
        linewidth=PLOT_CONFIG['grid_linewidth']  # Spessore linee (0.5)
    )

    ax2.set_xlabel('Ground Truth', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (Pred - True)', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Analysis', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Titolo generale con metriche
    if nmae_std is not None and nrmse_std is not None:
        metrics_str = f'NMAE: {nmae_value:.4f} ± {nmae_std:.4f} | NRMSE: {nrmse_value:.4f} ± {nrmse_std:.4f}'
    else:
        metrics_str = f'NMAE: {nmae_value:.4f} | NRMSE: {nrmse_value:.4f}'

    fig.suptitle(f'{title}\n{metrics_str}', fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Salva grafico su file se richiesto
    if save_path:
        # DPI centralizzata da constants per qualità consistente
        plt.savefig(save_path, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        print(f"✓ Scatter + Residuals plot salvato: {save_path}")

    return fig, axes


def plot_kfold_scatter(
    y_true_folds: list,
    y_pred_folds: list,
    fold_names: Optional[list] = None,
    title: str = "K-Fold Cross Validation",
    save_path: Optional[Path] = None,
    figsize: tuple = (18, 12),
    metrics_aggregated: Optional[Dict] = None
):
    """
    Crea griglia di scatter plot per K-fold cross validation

    Args:
        y_true_folds: Lista di array ground truth (uno per fold)
        y_pred_folds: Lista di array predictions (uno per fold)
        fold_names: Nomi dei fold (default: "Fold 1", "Fold 2", ...)
        title: Titolo principale
        save_path: Path dove salvare
        figsize: Dimensione figura
        metrics_aggregated: Metriche aggregate {metric: {'mean': ..., 'std': ...}}
    """
    n_folds = len(y_true_folds)

    # Determina layout griglia
    if n_folds <= 3:
        nrows, ncols = 1, n_folds
    elif n_folds <= 6:
        nrows, ncols = 2, 3
    else:
        nrows = int(np.ceil(n_folds / 3))
        ncols = 3

    # Fold names
    if fold_names is None:
        fold_names = [f'Fold {i+1}' for i in range(n_folds)]

    # Crea figura
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_folds == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot ogni fold con configurazione centralizzata
    # Ogni subplot mostra scatter plot ground truth vs predictions per un singolo fold
    # Metriche calcolate individualmente per confrontare performance tra fold
    for i, (y_true, y_pred, fold_name) in enumerate(zip(y_true_folds, y_pred_folds, fold_names)):
        ax = axes[i]

        # Scatter plot con stessa configurazione di plot_scatter_predictions()
        ax.scatter(
            y_true, y_pred,
            alpha=PLOT_CONFIG['scatter_alpha'],      # Trasparenza punti (0.95)
            s=PLOT_CONFIG['scatter_size'],           # Dimensione punti (40)
            color=PLOT_CONFIG['scatter_color'],      # Colore principale (#1E5F8C)
            edgecolors=PLOT_CONFIG['edge_color'],    # Colore bordo (darkblue)
            linewidth=PLOT_CONFIG['edge_linewidth']  # Spessore bordo (0.6)
        )

        # Calcola range assi: min >= 0 per dati energetici
        min_val = max(0, min(y_true.min(), y_pred.min()))
        max_val = max(y_true.max(), y_pred.max())

        # Linea ideale (y = x): predizione perfetta
        ax.plot(
            [min_val, max_val], [min_val, max_val],
            'r--',
            lw=1.5,  # Leggermente più sottile per subplot piccoli
            alpha=PLOT_CONFIG['ideal_line_alpha']  # Trasparenza (0.7)
        )

        # Imposta limiti assi espliciti per aspect ratio 1:1
        ax.set_xlim((min_val, max_val))
        ax.set_ylim((min_val, max_val))

        # Calcola metriche per questo fold individualmente
        # Utile per confrontare variabilità performance tra fold
        nmae_val = nmae(y_true, y_pred)
        nrmse_val = nrmse(y_true, y_pred)

        # Griglia di sfondo
        ax.grid(
            True,
            alpha=PLOT_CONFIG['grid_alpha'],         # Trasparenza griglia (0.3)
            linestyle='--',
            linewidth=PLOT_CONFIG['grid_linewidth']  # Spessore linee (0.5)
        )

        ax.set_xlabel('Ground Truth', fontsize=10, fontweight='bold')
        ax.set_ylabel('Prediction', fontsize=10, fontweight='bold')
        ax.set_title(f'{fold_name}\nNMAE: {nmae_val:.4f} | NRMSE: {nrmse_val:.4f}',
                     fontsize=11, fontweight='bold')

        # Aspect ratio uguale per mantenere proporzioni corrette
        ax.set_aspect('equal', adjustable='box')

    # Nascondi assi inutilizzati nella griglia
    for i in range(n_folds, len(axes)):
        axes[i].axis('off')

    # Titolo principale con metriche aggregate (procedura corretta per K-fold)
    # METODOLOGIA CORRETTA:
    # 1. Concatena tutti gli array di fold: [fold1, fold2, ..., foldN]
    # 2. Calcola metriche UNA SOLA VOLTA sull'array concatenato
    # NON fare media delle metriche per-fold (approccio errato)
    if metrics_aggregated is not None:
        # Metriche calcolate su array concatenato di tutti i fold
        nmae_val = metrics_aggregated.get('nmae', 0)
        nrmse_val = metrics_aggregated.get('nrmse', 0)
        nstd_val = metrics_aggregated.get('nstd', None)  # NSTD: std errori normalizzata

        # Formato titolo con NSTD se disponibile (cross-validation)
        if nstd_val is not None:
            metrics_str = f'Concatenated: NMAE: {nmae_val:.4f} | NRMSE: {nrmse_val:.4f} | NSTD: {nstd_val:.4f}'
        else:
            metrics_str = f'Concatenated: NMAE: {nmae_val:.4f} | NRMSE: {nrmse_val:.4f}'

        full_title = f'{title}\n{metrics_str}'
    else:
        full_title = title

    fig.suptitle(full_title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Salva grafico su file se richiesto
    if save_path:
        # DPI centralizzata da constants per qualità consistente
        plt.savefig(save_path, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        print(f"✓ K-Fold scatter plot salvato: {save_path}")

    return fig, axes
