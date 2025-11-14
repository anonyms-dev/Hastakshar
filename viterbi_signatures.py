#!/usr/bin/env python3
"""
viterbi_signatures.py

Clean pipeline:
 - load 2D signatures (x,y)
 - extract features (dx,dy,speed,angle)
 - train GaussianHMM on multiple sequences (uses lengths)
 - decode each test signature with Viterbi
 - plot each signature: left = XY colored by Viterbi states, right = state timeline
"""

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# -----------------------
# I/O / Feature helpers
# -----------------------
def load_signature_xy(path, usecols=(0,1), delimiter=','):
    """
    Load a signature file containing at least two columns (x, y).
    delimiter=None lets numpy infer whitespace/comma.
    """
    data = np.loadtxt(path, delimiter=delimiter, usecols=usecols)
    if data.ndim == 1:
        data = data.reshape(-1, len(usecols))
    return data[:, :2]

def extract_features(xy):
    """
    Given (T,2) xy coordinates return features (T-1, 4):
    [dx, dy, speed, angle]
    """
    deltas = np.diff(xy, axis=0)
    speed = np.linalg.norm(deltas, axis=1)
    angle = np.arctan2(deltas[:, 1], deltas[:, 0])
    feats = np.hstack((deltas, speed[:, None], angle[:, None]))
    return feats

# -----------------------
# Training
# -----------------------
def train_hmm_from_sequences(feature_seqs, n_states=8, cov_type='diag', n_iter=500, random_state=0):
    lengths = [len(s) for s in feature_seqs]
    X = np.vstack(feature_seqs)
    model = GaussianHMM(n_components=n_states, covariance_type=cov_type,
                        n_iter=n_iter, random_state=random_state, verbose=False)
    model.fit(X, lengths)
    return model

# -----------------------
# Plotting helpers
# -----------------------
def plot_single_viterbi(signature_xy, hidden_states, logprob=None, ax_traj=None, ax_time=None, cmap='tab20'):
    """
    Plot 1) XY trajectory colored by state and 2) timeline of states.
    signature_xy: (T,2) raw XY coordinates
    hidden_states: length T-1 (features length) or length T (if you used shifting).
      NOTE: We assume hidden_states aligns with points after first diff: i -> segment between point i and i+1.
      We will plot segments using hidden_states[i] for segment between signature_xy[i] and signature_xy[i+1].
    """
    T = len(signature_xy)
    # create axes if not provided
    if ax_traj is None or ax_time is None:
        fig, (ax_traj, ax_time) = plt.subplots(1, 2, figsize=(10, 4))

    # Trajectory: color each segment by its Viterbi state
    for i in range(1, T):
        # choose state for segment i-1 (since features are diff of points)
        st = hidden_states[i-1] if i-1 < len(hidden_states) else hidden_states[-1]
        ax_traj.plot(signature_xy[i-1:i+1, 0], signature_xy[i-1:i+1, 1],
                     color=plt.cm.jet(hidden_states[i-1] / max(hidden_states)), linewidth=1)
    title = "2D Signature with Viterbi States"
    if logprob is not None:
        title = f"{title}  (logprob={logprob:.2f})"
    ax_traj.set_title(title)
    ax_traj.set_xlabel("X"); ax_traj.set_ylabel("Y")
    ax_traj.axis('equal')
    ax_traj.invert_yaxis()

    # Timeline (states vs time)
    ax_time.step(np.arange(len(hidden_states)), hidden_states, where='mid')
    ax_time.set_title("Viterbi State Timeline")
    ax_time.set_xlabel("Segment index (time)")
    ax_time.set_ylabel("State ID")
    return ax_traj, ax_time

def plot_multiple_viterbi(signatures_xy, features, model, titles=None, n_cols=3, save_dir=None):
    """
    signatures_xy: list of (T_i,2)
    features: list of (T_i-1, F) matching signatures_xy
    model: trained HMM
    """
    n = len(signatures_xy)
    if n == 0:
        print("No signatures to plot.")
        return

    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols*2, figsize=(6*n_cols, 4*n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols*2)

    for idx, (xy, feat) in enumerate(zip(signatures_xy, features)):
        row, col = divmod(idx, n_cols)
        ax_traj = axes[row, col*2]
        ax_time = axes[row, col*2 + 1]

        logprob, states = model.decode(feat, algorithm="viterbi")
        title = titles[idx] if titles else f"Sig {idx+1}"
        plot_single_viterbi(xy, states, logprob=logprob, ax_traj=ax_traj, ax_time=ax_time)
        ax_traj.set_title(f"{title}\nlogprob={logprob:.1f}", fontsize=9)

    # Hide unused subplots
    total_plots = n_rows * n_cols * 2
    for i in range(n, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        # hide pair of axes
        axes[r, c*2].axis('off')
        axes[r, c*2+1].axis('off')

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "viterbi_grid_english.png")
        plt.savefig(out, dpi=150)
        print(f"Saved grid to {out}")
    plt.show()
    # plt.savefig('vitterbi_state_transition_english.jpg')

def plot_boxplots_of_features(features, titles=None, save_path=None):
    """
    features: list of (T_i-1, 4) arrays [dx, dy, speed, angle]
    Plots boxplots of each feature across all test signatures.
    """
    # stack all sequences
    all_feats = np.vstack(features)

    labels = ["dx", "dy", "speed", "angle"]

    plt.figure(figsize=(8, 6))
    plt.boxplot([all_feats[:, i] for i in range(all_feats.shape[1])],
                labels=labels, patch_artist=True)

    plt.title("Boxplot of Test Signature Features")
    plt.ylabel("Value")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved boxplot to {save_path}")

    plt.show()


# -----------------------
# main example usage
# -----------------------
def main():
    # ====== edit these paths ======
    genuine_folder = "/Users/balaji.raok/Documents/data_base/Hindi/Balaji1/"   # set to your folder of genuine signatures (x,y per row)
    test_folder = "/Users/balaji.raok/Documents/data_base/Hindi/Balaji1/"         # folder of signatures to visualize (genuine or forgery)
    file_glob = "*.txt"                    # pattern
    # ==============================

    # sanity: quick check that folders exist
    if not os.path.isdir(genuine_folder):
        raise SystemExit(f"genuine_folder '{genuine_folder}' not found. Edit the path in the script.")
    if not os.path.isdir(test_folder):
        raise SystemExit(f"test_folder '{test_folder}' not found. Edit the path in the script.")

    # ---- load training (genuine) signatures ----
    genuine_files = sorted(glob(os.path.join(genuine_folder, file_glob)))
    if len(genuine_files) == 0:
        raise SystemExit("No training files found in genuine_folder.")
    # Optionally sample or use all
    genuine_files = genuine_files[::20]  # e.g. genuine_files[::3] to downsample

    train_xy = [load_signature_xy(p) for p in genuine_files]
    train_feats = [extract_features(xy) for xy in train_xy]

    # ---- train HMM ----
    n_states = 12
    print(f"Training HMM on {len(train_feats)} sequences, total frames = {sum(len(f) for f in train_feats)}")
    hmm = train_hmm_from_sequences(train_feats, n_states=n_states, cov_type='diag', n_iter=500)

    # ---- load test signatures and decode/plot ----
    test_files = sorted(glob(os.path.join(test_folder, file_glob)))
    test_files = test_files[::20]
    if len(test_files) == 0:
        print("No test files found; using training examples as test.")
        test_files = genuine_files[:6]

    test_xy = [load_signature_xy(p) for p in test_files]
    test_feats = [extract_features(xy) for xy in test_xy]

    titles = [os.path.basename(p) for p in test_files]
    plot_boxplots_of_features(test_feats,
    save_path='/Users/balaji.raok/Documents/CVIP_2025/test_features_boxplot_hindi.png')
    # plot_multiple_viterbi(test_xy, test_feats, hmm, titles=titles, n_cols=3, save_dir='/Users/balaji.raok/Documents/CVIP_2025')

if __name__ == "__main__":
    main()
