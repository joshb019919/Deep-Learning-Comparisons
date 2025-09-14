from typing import List, Optional
from numpy import float64
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(
        train_losses: List[float64],
        val_losses: Optional[List[float64]],
        title="Loss Curves",
        xlabel="Iteration",
        ylabel="Loss",
        save_path: Optional[str] = None,
    ) -> None:
    """Plot training and optional validation loss curves."""
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss", lw=2)
    
    if val_losses:
        plt.plot(val_losses, label="Validation Losses", lw=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.close()
