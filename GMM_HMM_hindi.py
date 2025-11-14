import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import numpy as np
import os
from glob import glob

def load_signature_hmm(file_path):
    data = np.loadtxt(file_path, delimiter=',', usecols=(0, 1))
    print("ddd", len(data))
    return data[:, :2]  # Only x, y for simplicity

def extract_features(points):
    deltas = np.diff(points, axis=0)
    speed = np.linalg.norm(deltas, axis=1)
    angle = np.arctan2(deltas[:, 1], deltas[:, 0])
    features = np.hstack((deltas, speed[:, None], angle[:, None]))
    return features

def train_hmm(sequences, n_states=6):
    lengths = [len(seq) for seq in sequences]
    X = np.vstack(sequences)
    model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=1000)
    model.fit(X, lengths)
    return model

def plot_state_sequence(signature, states, title="State Transitions"):
    plt.figure(figsize=(6, 4))

    # a=[1:1:10]
    for i in range(1, len(signature)):
        # plt.subplot(2, 5, )
        plt.plot(signature[i-1:i+1, 0], signature[i-1:i+1, 1], color=plt.cm.jet(states[i] / max(states)))
    plt.title(title)
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), label="HMM State")
    plt.savefig('hmm_state_transition_1.jpg')
    plt.show()

def plot_viterbi_signature(signature, states, logprob=None):
    plt.figure(figsize=(6, 6))
    for i in range(1, len(signature)):
        plt.plot(signature[i-1:i+1, 0], signature[i-1:i+1, 1],
                 color=plt.cm.tab20(states[i] % 20), linewidth=2)
    if logprob is not None:
        plt.title(f"2D Signature with Viterbi States\nlogprob={logprob:.2f}")
    else:
        plt.title("2D Signature with Viterbi States")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()


def plot_multiple_state_sequences(signatures, states_list, titles, n_cols=3, save_path=None):
    n = len(signatures)
    print(n)
    # exit()
    n_rows = (n + n_cols - 1) // n_cols
    print(n_rows, n_cols)
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for idx, (signature, states, title) in enumerate(zip(signatures, states_list, titles)):
        plt.subplot(n_rows, n_cols, idx + 1)

        n_states = len(np.unique(states))
        for i in range(1, len(signature)):
            plt.scatter(signature[i-1:i+1, 0], signature[i-1:i+1, 1],
                     color=plt.cm.jet(states[i-1] / max(states)))

        plt.title(title, fontweight='bold')
        plt.axis('equal')
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        # plt.xlim(plt.xlim()[::-1])
        plt.ylim(plt.ylim()[::-1])

    plt.tight_layout()
    plt.savefig('hmm_state_transition_hindi.jpg')
    # if save_path:
    #     plt.savefig(save_path)
    plt.show()

def plot_viterbi_signature(signature, states, logprob=None):
    plt.figure(figsize=(6, 6))
    for i in range(1, len(signature)):
        plt.plot(signature[i-1:i+1, 0], signature[i-1:i+1, 1],
                 color=plt.cm.tab20(states[i] % 20), linewidth=2)
    if logprob is not None:
        plt.title(f"2D Signature with Viterbi States\nlogprob={logprob:.2f}")
    else:
        plt.title("2D Signature with Viterbi States")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_viterbi_states(signature, states, logprob, title="Viterbi State Sequence"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- (1) XY trajectory colored by Viterbi states ---
    for i in range(1, len(signature)):
        axes[0].plot(signature[i-1:i+1, 0], signature[i-1:i+1, 1],
                     color=plt.cm.tab20(states[i] % 20))  # cycle colors
    axes[0].set_title(f"{title}\n(logprob={logprob:.2f})", fontweight="bold")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].axis("equal")

    # --- (2) Timeline of states ---
    axes[1].plot(states, drawstyle="steps-mid")
    axes[1].set_title("Viterbi State Timeline")
    axes[1].set_xlabel("Time (points)")
    axes[1].set_ylabel("State ID")

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def plot_multiple_viterbi(signatures, features, model, titles=None, n_cols=2):
    """
    signatures: list of raw XY arrays
    features: list of feature arrays (after extract_features)
    model: trained HMM
    titles: optional list of titles
    n_cols: number of signature rows, each with 2 plots (traj + timeline)
    """
    n = len(signatures)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols*2, figsize=(8*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes]  # ensure iterable when only 1 row
    axes = np.array(axes).reshape(n_rows, n_cols*2)

    for idx, (sig, feat) in enumerate(zip(signatures, features)):
        row, col = divmod(idx, n_cols)
        ax_traj = axes[row, col*2]
        ax_time = axes[row, col*2 + 1]

        # Decode with Viterbi
        logprob, states = model.decode(feat, algorithm="viterbi")

        # --- Trajectory plot ---
        for i in range(1, len(sig)):
            ax_traj.plot(sig[i-1:i+1, 0], sig[i-1:i+1, 1],
                         color=plt.cm.tab20(states[i-1] % 20))
        ttl = titles[idx] if titles else f"Sig {idx+1}"
        ax_traj.set_title(f"{ttl}\nlogprob={logprob:.1f}", fontweight="bold")
        ax_traj.set_xlabel("X")
        ax_traj.set_ylabel("Y")
        ax_traj.axis("equal")

        # --- Timeline plot ---
        ax_time.plot(states, drawstyle="steps-mid")
        ax_time.set_title("Viterbi Timeline")
        ax_time.set_xlabel("Time")
        ax_time.set_ylabel("State")

    plt.tight_layout()
    plt.show()





def main():
    # === Load a few genuine signatures for training ===
    # user_folder = "SVC2004/user_1/"
    genuine_path = "/Users/balaji.raok/Documents/data_base/English/Balaji/"
    forgery_path = "/Users/balaji.raok/Documents/data_base/Hindi/Balaji/"

    # files = ["genuine_01.txt", "genuine_02.txt", "genuine_03.txt"]
    genuine_files = sorted(glob(os.path.join(genuine_path, "*.txt")))
    forgery_files = sorted(glob(os.path.join(forgery_path, "*.txt")))
    forgery_files = forgery_files[::5]
    # print(genuine_files)
    # exit()
    genuine_files = genuine_files[::5]
    sequences = [extract_features(load_signature_hmm(f)) for f in genuine_files]
    # print(sequences[0].shape)
    print(genuine_files)

    # === Train the HMM ===
    hmm = train_hmm(sequences, n_states=16)

    # === Load a test signature ===
    # test_path = forgery_files#user_folder + "genuine_04.txt"
    # print(test_path)
    # # exit()
    # test_raw = load_signature_hmm(test_path[1])
    # test_feat = extract_features(test_raw)
    # # print("pp",te)

    # # === Predict state sequence ===
    # hidden_states = hmm.predict(test_feat)
    # print("ppp", hidden_states[12])
    # print(test_raw)

    # # === Plot signature with color-coded states ===
    # plot_state_sequence(test_raw[1:], hidden_states, title="HMM State Transitions")
    test_files = genuine_files  # or use forgery_files[:6]
    test_signatures = [load_signature_hmm(f) for f in test_files]
    test_features = [extract_features(s) for s in test_signatures]
    state_sequences = [hmm.predict(f) for f in test_features]
    # titles = [os.path.basename(f) for f in test_files]
    # plot_multiple_viterbi(test_signatures, test_features, hmm, titles=titles, n_cols=3)

    # print(np.shape(test_features))
    logprob, hidden_states = hmm.decode(test_signatures[::5], algorithm="viterbi")
    # plot_viterbi_states(test_features[::25], hidden_states, logprob)
    # plot_viterbi_signature(X1, states, logprob)


    # plt.figure(figsize=(10,3))
    # plt.plot(state_sequences[0])
    # plt.title("Viterbi states across all sequences (concatenated)")
    # plt.show()


    # titles = [os.path.basename(f) for f in test_files]
    plot_multiple_state_sequences(test_signatures, state_sequences, titles, n_cols=5)

if __name__ == "__main__":
    main()
