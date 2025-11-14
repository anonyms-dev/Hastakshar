import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
'''
def generate_playcard_from_timeseries(data, x_col=0, y_col=1, title="Trajectory Card", save_path="card.png"):
    """
    Generate a play-card style image from 2D trajectory (time series).
    
    data : np.ndarray
        Time-series data (rows = time, columns = features).
    x_col, y_col : int
        Which columns to use for X and Y plotting.
    title : str
        Title at the top of the card.
    save_path : str
        Output image path.
    """
    x = data[:, x_col]
    y = data[:, y_col]

    fig, ax = plt.subplots(figsize=(4,6), dpi=150)

    # Card shape
    card = FancyBboxPatch((0,0), 1, 1,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          edgecolor="black", facecolor="white",
                          linewidth=2, transform=ax.transAxes, zorder=0)
    ax.add_patch(card)

    # Plot trajectory
    ax.plot(x, y, 'b-', linewidth=1.5, zorder=1)

    # Title
    ax.text(0.5, 1.02, title, fontsize=12, ha="center", va="bottom", transform=ax.transAxes)

    # Formatting
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.show()
    print(f"Card saved at: {save_path}")


# ==== Example usage ====
# Suppose your data is in a file "timeseries.txt" with tab-separated values:
data = np.loadtxt("/Users/balaji.raok/Documents/Digital_smell/Dataset/dataset_cofee/LQ_Coffee/LQ_02.txt")  # replace with your data file

# Generate card with first 2 columns as X and Y
generate_playcard_from_timeseries(data, x_col=0, y_col=1, title="Signature A", save_path="signature_card.png")
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
from glob import glob

def generate_card(ax, x, y, title=""):
    """Draws a play card on given axis."""
    # Card shape
    card = FancyBboxPatch((0,0), 1, 1,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          edgecolor="black", facecolor="white",
                          linewidth=2, transform=ax.transAxes, zorder=0)
    ax.add_patch(card)

    # Trajectory
    ax.plot(x, y, 'b-', linewidth=1.5, zorder=1)

    # Title
    ax.text(0.5, 1.02, title, fontsize=10,
            ha="center", va="bottom", transform=ax.transAxes)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()

def generate_deck(folder, file_pattern="*.txt", n_cols=4, save_path="deck.png"):
    files = sorted(glob(os.path.join(folder, file_pattern)))
    n = len(files)
    if n == 0:
        print("No files found.")
        return
    
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, f in enumerate(files):
        data = np.loadtxt(f)
        x, y = data[:,0], data[:,1]  # col0=X, col1=Y
        generate_card(axes[i], x, y, title=os.path.basename(f))

    # hide unused slots
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Deck saved at {save_path}")


# ===== Example Usage =====
generate_deck("/Users/balaji.raok/Documents/Digital_smell/Dataset/dataset_cofee/LQ_Coffee", file_pattern="*.txt", n_cols=4)
