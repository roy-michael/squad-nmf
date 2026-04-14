"""
Visualization and diagnostics module for underwater vehicle sound classification.

Provides publication-quality plotting utilities for spectrograms, NMF components,
feature distributions, and classification results.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.decomposition import PCA
import warnings


def plot_spectrogram(
    S_mel: np.ndarray,
    sr: int,
    hop_length: int,
    title: str = "",
    save_path: Optional[str] = None,
    cmap: str = "magma",
    top_db: float = 80.0,
) -> plt.Figure:
    """
    Plot a Mel-scaled spectrogram with time and frequency axes.

    Parameters
    ----------
    S_mel : np.ndarray
        Mel-scaled spectrogram (n_mels × n_frames).
    sr : int
        Sample rate in Hz.
    hop_length : int
        Number of samples per frame.
    title : str, optional
        Title for the plot (default: "").
    save_path : str, optional
        Path to save the figure. If None, figure is not saved (default: None).
    cmap : str, optional
        Colormap name (default: "magma"). Options: 'viridis', 'plasma', 'magma', etc.
    top_db : float, optional
        Dynamic range in dB for log scaling (default: 80.0).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle for further manipulation.

    Raises
    ------
    ValueError
        If input dimensions are invalid or parameters are out of range.

    Examples
    --------
    >>> import numpy as np
    >>> S_mel = np.random.rand(128, 200)  # 128 mel bins, 200 frames
    >>> fig = plot_spectrogram(S_mel, sr=16000, hop_length=512, title="Spectrogram")
    >>> plt.show()
    """
    if S_mel.ndim != 2:
        raise ValueError(f"S_mel must be 2D, got shape {S_mel.shape}")
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    if top_db <= 0:
        raise ValueError(f"top_db must be positive, got {top_db}")

    # Convert to log scale (dB)
    S_db = 10 * np.log10(np.maximum(S_mel, 1e-10))
    S_db_normalized = np.maximum(S_db - np.max(S_db), -top_db)

    n_mels, n_frames = S_mel.shape

    # Time axis in seconds
    times = np.arange(n_frames) * hop_length / sr

    # Frequency axis in Hz (approximate)
    freqs = np.linspace(0, sr / 2, n_mels)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Plot spectrogram
    im = ax.pcolormesh(
        times,
        freqs,
        S_db_normalized,
        shading="auto",
        cmap=cmap,
        rasterized=True,
    )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_title(title or "Mel Spectrogram", fontsize=12, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label="Magnitude (dB)")

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_nmf_components(
    W: np.ndarray,
    H: np.ndarray,
    freqs_hz: Optional[np.ndarray] = None,
    times: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
    log_freq: bool = False,
) -> plt.Figure:
    """
    Plot NMF basis functions (W) and activations (H).

    Left column shows basis functions with frequency axis, right column shows
    activations with time axis. Includes peak frequency annotations on basis functions.

    Parameters
    ----------
    W : np.ndarray
        NMF basis functions (n_mels × n_components).
    H : np.ndarray
        NMF activations (n_components × n_frames).
    freqs_hz : np.ndarray, optional
        Frequency values for basis functions (n_mels,).
        If None, uses arbitrary frequency bins.
    times : np.ndarray, optional
        Time values for activations (n_frames,).
        If None, uses arbitrary frame indices.
    title : str, optional
        Title for the plot (default: "").
    save_path : str, optional
        Path to save the figure. If None, figure is not saved (default: None).
    log_freq : bool, optional
        Use logarithmic frequency scale (default: False).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle for further manipulation.

    Raises
    ------
    ValueError
        If input dimensions are invalid or mismatched.

    Examples
    --------
    >>> W = np.random.rand(128, 6)  # 6 components, 128 frequency bins
    >>> H = np.random.rand(6, 100)  # 100 time frames
    >>> fig = plot_nmf_components(W, H)
    >>> plt.show()
    """
    if W.ndim != 2 or H.ndim != 2:
        raise ValueError(f"W and H must be 2D, got W{W.shape} and H{H.shape}")
    if W.shape[1] != H.shape[0]:
        raise ValueError(
            f"W and H component mismatch: W has {W.shape[1]} components "
            f"but H has {H.shape[0]}"
        )

    n_mels, n_components = W.shape
    n_frames = H.shape[1]

    # Default frequency and time axes
    if freqs_hz is None:
        freqs_hz = np.arange(n_mels)
    if times is None:
        times = np.arange(n_frames)

    if len(freqs_hz) != n_mels:
        raise ValueError(
            f"freqs_hz length mismatch: expected {n_mels}, got {len(freqs_hz)}"
        )
    if len(times) != n_frames:
        raise ValueError(
            f"times length mismatch: expected {n_frames}, got {len(times)}"
        )

    fig, axes = plt.subplots(
        n_components, 2, figsize=(14, 2.5 * n_components), dpi=150
    )

    # Handle single component case
    if n_components == 1:
        axes = axes.reshape(1, -1)

    for c in range(n_components):
        # Plot basis function (W)
        ax_w = axes[c, 0]
        ax_w.plot(W[:, c], freqs_hz, linewidth=2, color="steelblue")
        ax_w.fill_betweenx(freqs_hz, 0, W[:, c], alpha=0.3, color="steelblue")
        ax_w.set_ylabel("Frequency (Hz)", fontsize=10)
        ax_w.set_xlabel("Amplitude", fontsize=10)
        ax_w.set_title(f"Basis {c+1}", fontsize=11, fontweight="bold")

        if log_freq and np.min(freqs_hz) > 0:
            ax_w.set_yscale("log")

        ax_w.grid(True, alpha=0.3, linestyle="--")

        # Annotate top 5 peaks
        peaks_idx = np.argsort(W[:, c])[-5:]
        for peak_idx in peaks_idx:
            if W[peak_idx, c] > 0:
                freq_val = freqs_hz[peak_idx]
                ax_w.annotate(
                    f"{freq_val:.0f}Hz",
                    xy=(W[peak_idx, c], freq_val),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

        # Plot activation (H)
        ax_h = axes[c, 1]
        ax_h.plot(times, H[c, :], linewidth=2, color="coral")
        ax_h.fill_between(times, 0, H[c, :], alpha=0.3, color="coral")
        ax_h.set_xlabel("Time", fontsize=10)
        ax_h.set_ylabel("Activation", fontsize=10)
        ax_h.set_title(f"Activation {c+1}", fontsize=11, fontweight="bold")
        ax_h.grid(True, alpha=0.3, linestyle="--")

        # Normalize y-axis to [0, 1]
        h_max = np.max(H[c, :])
        if h_max > 0:
            ax_h.set_ylim([0, h_max * 1.1])

    fig.suptitle(title or "NMF Components", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: Dict[int, str],
    title: str = "",
    save_path: Optional[str] = None,
    use_pca: bool = True,
) -> plt.Figure:
    """
    Plot feature distribution and dimensionality reduction visualization.

    Creates a 2x2 grid: feature statistics (box plot, violin plot) and
    dimensionality reduction (PCA 2D projection).

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples × n_features).
    labels : np.ndarray
        Class labels (n_samples,).
    label_names : Dict[int, str]
        Mapping of label indices to class names.
    title : str, optional
        Title for the plot (default: "").
    save_path : str, optional
        Path to save the figure. If None, figure is not saved (default: None).
    use_pca : bool, optional
        Use PCA for dimensionality reduction (default: True).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle for further manipulation.

    Raises
    ------
    ValueError
        If input dimensions are invalid or no samples per class.

    Examples
    --------
    >>> features = np.random.rand(100, 20)
    >>> labels = np.repeat([0, 1, 2], [40, 35, 25])
    >>> label_names = {0: "Class0", 1: "Class1", 2: "Class2"}
    >>> fig = plot_feature_distribution(features, labels, label_names)
    >>> plt.show()
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if len(features) != len(labels):
        raise ValueError(
            f"features and labels length mismatch: {len(features)} vs {len(labels)}"
        )

    unique_labels = np.unique(labels)
    if len(unique_labels) == 0:
        raise ValueError("No samples in dataset")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # Prepare data for plotting
    classes = sorted(unique_labels)
    colors = sns.color_palette("husl", len(classes))
    color_map = {cls: colors[i] for i, cls in enumerate(classes)}

    # 1. Box plot of first principal component per class
    ax = axes[0, 0]
    if use_pca and features.shape[1] > 1:
        pca = PCA(n_components=min(features.shape[1], 3))
        features_pca = pca.fit_transform(features)
        feature_for_plot = features_pca[:, 0]
        explained_var = pca.explained_variance_ratio_[0]
        ax.set_ylabel(f"PC1 ({explained_var*100:.1f}%)", fontsize=10)
    else:
        feature_for_plot = features[:, 0] if features.shape[1] > 0 else features.flatten()
        ax.set_ylabel("Feature 1", fontsize=10)

    box_data = [feature_for_plot[labels == cls] for cls in classes]
    bp = ax.boxplot(
        box_data,
        labels=[label_names.get(cls, str(cls)) for cls in classes],
        patch_artist=True,
    )
    for patch, cls in zip(bp["boxes"], classes):
        patch.set_facecolor(color_map[cls])
    ax.set_title("Feature Distribution (Box Plot)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # 2. Violin plot
    ax = axes[0, 1]
    parts = ax.violinplot(
        box_data,
        positions=range(len(classes)),
        showmeans=True,
        showmedians=True,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(color_map[classes[i]])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([label_names.get(cls, str(cls)) for cls in classes])
    ax.set_ylabel("Feature Value", fontsize=10)
    ax.set_title("Feature Distribution (Violin Plot)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # 3. PCA 2D projection
    ax = axes[1, 0]
    if use_pca and features.shape[1] > 1:
        pca = PCA(n_components=min(features.shape[1], 2))
        features_pca = pca.fit_transform(features)
        for cls in classes:
            mask = labels == cls
            ax.scatter(
                features_pca[mask, 0],
                features_pca[mask, 1],
                label=label_names.get(cls, str(cls)),
                color=color_map[cls],
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
    else:
        for cls in classes:
            mask = labels == cls
            ax.scatter(
                features[mask, 0],
                features[mask, 1] if features.shape[1] > 1 else np.zeros_like(features[mask, 0]),
                label=label_names.get(cls, str(cls)),
                color=color_map[cls],
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
        ax.set_xlabel("Feature 1", fontsize=10)
        ax.set_ylabel("Feature 2", fontsize=10)

    ax.set_title("PCA Projection (2D)", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # 4. Feature statistics table
    ax = axes[1, 1]
    ax.axis("off")

    stats_text = "Feature Statistics per Class:\n\n"
    for cls in classes:
        mask = labels == cls
        class_features = features[mask]
        mean_feat = np.mean(class_features, axis=0).mean()
        std_feat = np.mean(np.std(class_features, axis=0))
        n_samples = np.sum(mask)

        stats_text += f"{label_names.get(cls, str(cls))}:\n"
        stats_text += f"  n_samples: {n_samples}\n"
        stats_text += f"  mean: {mean_feat:.4f}\n"
        stats_text += f"  std: {std_feat:.4f}\n\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    fig.suptitle(
        title or "Feature Distribution Analysis",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Dict[int, str],
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Shows both normalized and unnormalized versions side by side with
    counts and percentages.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels (n_samples,).
    y_pred : np.ndarray
        Predicted class labels (n_samples,).
    label_names : Dict[int, str]
        Mapping of label indices to class names.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved (default: None).
    normalize : bool, optional
        Include normalized version in plot (default: True).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle for further manipulation.

    Raises
    ------
    ValueError
        If input dimensions are invalid or mismatched.

    Examples
    --------
    >>> y_true = np.array([0, 1, 2, 0, 1, 2])
    >>> y_pred = np.array([0, 1, 1, 0, 1, 2])
    >>> label_names = {0: "A", 1: "B", 2: "C"}
    >>> fig = plot_confusion_matrix(y_true, y_pred, label_names)
    >>> plt.show()
    """
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1D, got {y_true.shape} and {y_pred.shape}")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred length mismatch: {len(y_true)} vs {len(y_pred)}"
        )

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    class_labels = [label_names.get(cls, str(cls)) for cls in classes]

    # Create figure with 1 or 2 subplots
    n_cols = 2 if normalize else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), dpi=150)
    if n_cols == 1:
        axes = [axes]

    # Plot unnormalized
    ax = axes[0]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix (Counts)", fontsize=12, fontweight="bold")

    # Plot normalized
    if normalize:
        ax = axes[1]
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Greens",
            cbar_kws={"label": "Percentage"},
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title("Confusion Matrix (Normalized)", fontsize=12, fontweight="bold")

    fig.suptitle("Classification Results", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: Dict[int, str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiclass classification (one-vs-rest).

    Shows individual class ROC curves, macro-averaged, and micro-averaged ROC curves
    with AUC scores.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels (n_samples,).
    y_proba : np.ndarray
        Predicted class probabilities (n_samples, n_classes).
    label_names : Dict[int, str]
        Mapping of label indices to class names.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved (default: None).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle for further manipulation.

    Raises
    ------
    ValueError
        If input dimensions are invalid or mismatched.

    Examples
    --------
    >>> y_true = np.array([0, 1, 2, 0, 1, 2])
    >>> y_proba = np.random.rand(6, 3)
    >>> y_proba /= y_proba.sum(axis=1, keepdims=True)  # Normalize to probabilities
    >>> label_names = {0: "A", 1: "B", 2: "C"}
    >>> fig = plot_roc_curves(y_true, y_proba, label_names)
    >>> plt.show()
    """
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")
    if y_proba.ndim != 2:
        raise ValueError(f"y_proba must be 2D, got shape {y_proba.shape}")
    if len(y_true) != len(y_proba):
        raise ValueError(
            f"y_true and y_proba length mismatch: {len(y_true)} vs {len(y_proba)}"
        )

    classes = sorted(np.unique(y_true))
    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError("Need at least 2 classes for ROC curve")

    if y_proba.shape[1] != n_classes:
        raise ValueError(
            f"y_proba column count mismatch: expected {n_classes}, got {y_proba.shape[1]}"
        )

    # Convert class labels to indices if necessary
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_true_idx = np.array([class_to_idx[y] for y in y_true])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Plot 1: Per-class ROC curves
    ax = axes[0]
    colors = sns.color_palette("husl", n_classes)

    for cls_idx, cls in enumerate(classes):
        # One-vs-rest binary classification
        y_true_binary = (y_true_idx == cls_idx).astype(int)
        y_pred_binary = y_proba[:, cls_idx]

        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            label=f"{label_names.get(cls, str(cls))} (AUC = {roc_auc:.3f})",
            color=colors[cls_idx],
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Plot 2: Macro and Micro averaged ROC
    ax = axes[1]

    # Macro-averaged
    all_fpr = np.linspace(0, 1, 100)
    tpr_macros = []

    for cls_idx, cls in enumerate(classes):
        y_true_binary = (y_true_idx == cls_idx).astype(int)
        y_pred_binary = y_proba[:, cls_idx]
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
        tpr_interp = np.interp(all_fpr, fpr, tpr)
        tpr_macros.append(tpr_interp)

    tpr_macro_avg = np.mean(tpr_macros, axis=0)
    macro_auc = auc(all_fpr, tpr_macro_avg)

    ax.plot(
        all_fpr,
        tpr_macro_avg,
        label=f"Macro-averaged (AUC = {macro_auc:.3f})",
        color="red",
        linewidth=2.5,
    )

    # Micro-averaged (global)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            micro_auc = roc_auc_score(y_true_idx, y_proba, multi_class="ovr", average="micro")
            ax.plot(
                [0, 1],
                [0, 1],
                label=f"Micro-averaged (AUC = {micro_auc:.3f})",
                color="blue",
                linewidth=2.5,
            )
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Averaged ROC Curves", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    fig.suptitle("ROC Curve Analysis", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Dict[int, str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-class classification metrics (precision, recall, F1).

    Creates bar charts showing per-class metrics and displays a text report
    with support (number of samples per class).

    Parameters
    ----------
    y_true : np.ndarray
        True class labels (n_samples,).
    y_pred : np.ndarray
        Predicted class labels (n_samples,).
    label_names : Dict[int, str]
        Mapping of label indices to class names.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved (default: None).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle for further manipulation.

    Raises
    ------
    ValueError
        If input dimensions are invalid or mismatched.

    Examples
    --------
    >>> y_true = np.array([0, 1, 2, 0, 1, 2])
    >>> y_pred = np.array([0, 1, 1, 0, 1, 2])
    >>> label_names = {0: "A", 1: "B", 2: "C"}
    >>> fig = plot_classification_report(y_true, y_pred, label_names)
    >>> plt.show()
    """
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1D, got {y_true.shape} and {y_pred.shape}")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred length mismatch: {len(y_true)} vs {len(y_pred)}"
        )

    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    class_labels = [label_names.get(cls, str(cls)) for cls in classes]

    # Compute metrics per class
    precisions = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Bar chart of metrics
    x = np.arange(len(classes))
    width = 0.25

    ax1.bar(x - width, precisions, width, label="Precision", alpha=0.8, color="steelblue")
    ax1.bar(x, recalls, width, label="Recall", alpha=0.8, color="coral")
    ax1.bar(x + width, f1s, width, label="F1-Score", alpha=0.8, color="seagreen")

    ax1.set_xlabel("Class", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Per-Class Metrics", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_labels)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Text report
    ax2.axis("off")

    report_str = classification_report(
        y_true, y_pred, labels=classes, target_names=class_labels, digits=3
    )

    # Add support info
    support_str = "\nSupport (samples per class):\n"
    for cls, label in zip(classes, class_labels):
        support = np.sum(y_true == cls)
        support_str += f"  {label}: {support}\n"

    full_text = "Classification Report\n" + "=" * 40 + "\n" + report_str + support_str

    ax2.text(
        0.05,
        0.95,
        full_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    fig.suptitle("Classification Metrics", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_waveform(
    y: np.ndarray,
    sr: int,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot audio waveform in time domain.

    Parameters
    ----------
    y : np.ndarray
        Audio time series (n_samples,).
    sr : int
        Sample rate in Hz.
    title : str, optional
        Title for the plot (default: "").
    save_path : str, optional
        Path to save the figure. If None, figure is not saved (default: None).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle for further manipulation.

    Raises
    ------
    ValueError
        If input dimensions are invalid or sample rate is invalid.

    Examples
    --------
    >>> y = np.random.randn(16000)  # 1 second at 16 kHz
    >>> fig = plot_waveform(y, sr=16000, title="Audio Waveform")
    >>> plt.show()
    """
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")

    # Time axis
    times = np.arange(len(y)) / sr

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)

    ax.plot(times, y, linewidth=0.5, color="steelblue", alpha=0.8)
    ax.fill_between(times, y, alpha=0.3, color="steelblue")

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.set_title(title or "Waveform", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_diagnostic_report(
    audio_path: str,
    pipeline,
    save_dir: str = "./diagnostics",
) -> Dict:
    """
    Create comprehensive diagnostic report for audio file.

    Generates waveform, spectrogram, NMF components, and classification
    predictions, saving individual plots and returning metadata.

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    pipeline : AudioClassificationPipeline
        Trained audio classification pipeline.
    save_dir : str, optional
        Directory to save diagnostic plots (default: "./diagnostics").

    Returns
    -------
    report : dict
        Dictionary with keys:
        - 'waveform_path': Path to waveform plot
        - 'spectrogram_path': Path to spectrogram plot
        - 'nmf_components_path': Path to NMF components plot
        - 'predicted_class': Predicted class ID
        - 'predicted_label': Predicted class name
        - 'confidence': Prediction confidence
        - 'probabilities': Class probabilities
        - 'audio_path': Input audio path
        - 'metadata': Additional metadata dict

    Raises
    ------
    RuntimeError
        If classifier is not trained or audio loading fails.
    FileNotFoundError
        If audio file or pipeline components do not exist.

    Examples
    --------
    >>> report = create_diagnostic_report("sample.wav", pipeline, save_dir="./diags")
    >>> print(f"Predicted: {report['predicted_label']}")
    >>> print(f"Files saved to: {report['metadata']['save_dir']}")
    """
    import os
    from audio_loader import load_audio

    if not pipeline.classifier.is_trained:
        raise RuntimeError("Classifier must be trained before diagnostics")

    os.makedirs(save_dir, exist_ok=True)

    # Load audio
    y, sr = load_audio(audio_path)

    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Plot waveform
    fig_wave = plot_waveform(y, sr, title=f"Waveform: {base_name}")
    waveform_path = os.path.join(save_dir, f"{base_name}_waveform.png")
    fig_wave.savefig(waveform_path, dpi=150, bbox_inches="tight")
    plt.close(fig_wave)

    # Preprocess
    mel_spec = pipeline.preprocessor.preprocess(y, sr)
    n_mels, n_frames = mel_spec.shape

    # Plot spectrogram
    hop_length = pipeline.preprocessor.hop_length
    fig_spec = plot_spectrogram(
        mel_spec, sr, hop_length, title=f"Mel Spectrogram: {base_name}"
    )
    spectrogram_path = os.path.join(save_dir, f"{base_name}_spectrogram.png")
    fig_spec.savefig(spectrogram_path, dpi=150, bbox_inches="tight")
    plt.close(fig_spec)

    # Extract NMF components
    W, H, _ = pipeline.feature_extractor.W, pipeline.feature_extractor.H, []
    if W is None or H is None:
        # Re-extract if not cached
        from feature_extractor import extract_nmf_features
        W, H, _ = extract_nmf_features(mel_spec, n_components=pipeline.feature_extractor.n_components)

    freqs_hz = np.linspace(0, sr / 2, n_mels)
    times = np.arange(n_frames) * hop_length / sr

    fig_nmf = plot_nmf_components(
        W, H, freqs_hz=freqs_hz, times=times, title=f"NMF Components: {base_name}"
    )
    nmf_path = os.path.join(save_dir, f"{base_name}_nmf_components.png")
    fig_nmf.savefig(nmf_path, dpi=150, bbox_inches="tight")
    plt.close(fig_nmf)

    # Classify
    prediction = pipeline.classify(audio_path)

    # Prepare report
    report = {
        "waveform_path": waveform_path,
        "spectrogram_path": spectrogram_path,
        "nmf_components_path": nmf_path,
        "predicted_class": prediction["predicted_class"],
        "predicted_label": prediction["predicted_label"],
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "audio_path": audio_path,
        "metadata": {
            "save_dir": save_dir,
            "sample_rate": sr,
            "duration_sec": len(y) / sr,
            "n_samples": len(y),
            "n_mels": n_mels,
            "n_frames": n_frames,
            "hop_length": hop_length,
        },
    }

    return report
