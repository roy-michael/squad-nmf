"""
Temporal clustering visualization - visualize component activations over time.

Provides visualizations for understanding how NMF components activate during
different temporal patterns, making it easy to see simultaneous vs sequential
sound sources.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


def plot_component_activation_timeline(
    H: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: Path,
    sample_rate: int = 8000,
    hop_length: int = 512,
    title: str = "Component Activation Timeline",
) -> None:
    """
    Create a timeline plot showing which components are active in each temporal cluster.

    This is the main visualization showing component activation over time.

    Args:
        H: NMF activation matrix (n_components × n_time_frames)
        cluster_labels: K-means cluster assignment per frame
        output_path: Where to save the plot
        sample_rate: Audio sample rate
        hop_length: Hop length used in STFT
        title: Plot title

    Returns:
        None (saves plot to disk)
    """
    n_components, n_frames = H.shape
    
    # Convert frame indices to time (seconds)
    time_seconds = np.arange(n_frames) * hop_length / sample_rate
    
    # Create figure with component activations and cluster assignments
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # ===== Plot 1: Component Activations (Stacked Area) =====
    ax1 = fig.add_subplot(gs[0])
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_components))
    ax1.stackplot(
        time_seconds,
        *H,
        labels=[f"Component {i}" for i in range(n_components)],
        colors=colors,
        alpha=0.7
    )
    
    ax1.set_ylabel("Activation Level", fontsize=11, fontweight="bold")
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", ncol=n_components, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time_seconds[-1])
    
    # ===== Plot 2: Individual Component Lines =====
    ax2 = fig.add_subplot(gs[1])
    
    for i in range(n_components):
        ax2.plot(
            time_seconds,
            H[i],
            label=f"Comp {i}",
            color=colors[i],
            linewidth=1.5,
            alpha=0.8
        )
    
    ax2.set_ylabel("Activation", fontsize=10, fontweight="bold")
    ax2.set_title("Individual Component Activations", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", ncol=n_components, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, time_seconds[-1])
    
    # ===== Plot 3: Temporal Cluster Assignments =====
    ax3 = fig.add_subplot(gs[2])
    
    n_clusters = len(np.unique(cluster_labels))
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        frame_indices = np.where(mask)[0]
        
        if len(frame_indices) > 0:
            start_frame = frame_indices[0]
            end_frame = frame_indices[-1]
            start_time = time_seconds[start_frame]
            end_time = time_seconds[end_frame]
            
            rect = mpatches.Rectangle(
                (start_time, -0.5),
                end_time - start_time,
                1.0,
                facecolor=cluster_colors[cluster_id],
                edgecolor="black",
                linewidth=1.5,
                alpha=0.7
            )
            ax3.add_patch(rect)
            
            # Label the cluster
            mid_time = (start_time + end_time) / 2
            ax3.text(
                mid_time,
                0,
                f"Pattern {cluster_id}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold"
            )
    
    ax3.set_ylim(-1, 1)
    ax3.set_xlim(0, time_seconds[-1])
    ax3.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Clusters", fontsize=10, fontweight="bold")
    ax3.set_title("Temporal Cluster Assignments", fontsize=11, fontweight="bold")
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved component activation timeline: {output_path}")
    plt.close()


def plot_component_heatmap(
    H: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: Path,
    sample_rate: int = 8000,
    hop_length: int = 512,
    title: str = "Component Activation Heatmap",
) -> None:
    """
    Create a heatmap showing component activations over time.

    Red = high activation, Blue = low activation. Easily see which
    components are active during each temporal pattern.

    Args:
        H: NMF activation matrix (n_components × n_time_frames)
        cluster_labels: K-means cluster assignment per frame
        output_path: Where to save the plot
        sample_rate: Audio sample rate
        hop_length: Hop length used in STFT
        title: Plot title

    Returns:
        None (saves plot to disk)
    """
    n_components, n_frames = H.shape
    
    # Convert frame indices to time
    time_seconds = np.arange(n_frames) * hop_length / sample_rate
    
    # Downsample H matrix for visualization (too many frames = illegible)
    downsample_factor = max(1, n_frames // 500)
    H_downsampled = H[:, ::downsample_factor]
    time_downsampled = time_seconds[::downsample_factor]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot heatmap
    im = ax.imshow(
        H_downsampled,
        aspect="auto",
        cmap="hot",
        origin="lower",
        interpolation="nearest"
    )
    
    # Set ticks and labels
    ax.set_ylabel("Component", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    
    # Convert frame indices to time for x-axis
    n_downsampled_frames = len(time_downsampled)
    x_ticks = np.linspace(0, n_downsampled_frames - 1, 10)
    x_labels = [f"{time_downsampled[int(i)]:.1f}" for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    
    # Y-axis: component indices
    ax.set_yticks(range(n_components))
    ax.set_yticklabels([f"Comp {i}" for i in range(n_components)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Activation Level")
    
    # Add cluster boundaries as vertical lines
    n_clusters = len(np.unique(cluster_labels))
    cluster_labels_downsampled = cluster_labels[::downsample_factor]
    
    for i in range(1, len(cluster_labels_downsampled)):
        if cluster_labels_downsampled[i] != cluster_labels_downsampled[i - 1]:
            ax.axvline(x=i, color="cyan", linewidth=2, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved component activation heatmap: {output_path}")
    plt.close()


def plot_pattern_summary(
    H: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: Path,
    sample_rate: int = 8000,
    hop_length: int = 512,
) -> None:
    """
    Create a summary plot showing mean activations for each pattern.

    Shows which components are strongest in each temporal pattern,
    making it easy to see differences between simultaneous vs sequential sources.

    Args:
        H: NMF activation matrix (n_components × n_time_frames)
        cluster_labels: K-means cluster assignment per frame
        output_path: Where to save the plot
        sample_rate: Audio sample rate
        hop_length: Hop length used in STFT

    Returns:
        None (saves plot to disk)
    """
    n_components = H.shape[0]
    n_clusters = len(np.unique(cluster_labels))
    
    # Calculate mean activation per component per cluster
    cluster_means = np.zeros((n_clusters, n_components))
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_means[cluster_id] = H[:, mask].mean(axis=1)
    
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5))
    
    if n_clusters == 1:
        axes = [axes]
    
    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        
        components = np.arange(n_components)
        activations = cluster_means[cluster_id]
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_components))
        bars = ax.bar(components, activations, color=colors, alpha=0.7, edgecolor="black")
        
        # Highlight active components (> 0.05)
        for i, (bar, activation) in enumerate(zip(bars, activations)):
            if activation > 0.05:
                bar.set_linewidth(3)
                bar.set_edgecolor("red")
        
        ax.set_xlabel("Component", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Activation", fontsize=11, fontweight="bold")
        ax.set_title(f"Pattern {cluster_id}\n(Mean activations)", fontsize=12, fontweight="bold")
        ax.set_xticks(components)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(cluster_means.max() * 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved pattern summary: {output_path}")
    plt.close()


def create_all_visualizations(
    H: np.ndarray,
    cluster_labels: np.ndarray,
    output_dir: Path,
    filename: str,
    sample_rate: int = 8000,
    hop_length: int = 512,
) -> Dict[str, Path]:
    """
    Create all temporal clustering visualizations for a single file.

    Args:
        H: NMF activation matrix (n_components × n_time_frames)
        cluster_labels: K-means cluster assignment per frame
        output_dir: Directory to save plots
        filename: Base filename for plots
        sample_rate: Audio sample rate
        hop_length: Hop length used in STFT

    Returns:
        Dictionary mapping plot type to output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = Path(filename).stem
    
    results = {}
    
    # Timeline plot
    timeline_path = output_dir / f"{stem}_timeline.png"
    plot_component_activation_timeline(
        H,
        cluster_labels,
        timeline_path,
        sample_rate=sample_rate,
        hop_length=hop_length,
        title=f"Component Activation Timeline - {stem}",
    )
    results["timeline"] = timeline_path
    
    # Heatmap
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    plot_component_heatmap(
        H,
        cluster_labels,
        heatmap_path,
        sample_rate=sample_rate,
        hop_length=hop_length,
        title=f"Component Activation Heatmap - {stem}",
    )
    results["heatmap"] = heatmap_path
    
    # Pattern summary
    summary_path = output_dir / f"{stem}_summary.png"
    plot_pattern_summary(
        H,
        cluster_labels,
        summary_path,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    results["summary"] = summary_path
    
    return results
