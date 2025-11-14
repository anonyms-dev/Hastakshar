import numpy as np
import matplotlib.pyplot as plt

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def multivariate_dtw(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean_dist(seq1[i - 1], seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # insertion
                dtw_matrix[i, j - 1],    # deletion
                dtw_matrix[i - 1, j - 1] # match
            )

    # Traceback path
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        directions = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
        i, j = min(directions, key=lambda x: dtw_matrix[x[0], x[1]])
    path.reverse()

    return dtw_matrix[n, m], dtw_matrix, path


# Two synthetic "signature-like" sequences (x, y) pairs
sig1 = [[0, 0], [1, 1], [2, 1], [3, 2], [4, 3]]
sig2 = [[0, 0], [1, 0.9], [2, 1.1], [3, 2.1], [4, 3.2], [5, 4]]

dist, matrix, path = multivariate_dtw(sig1, sig2)
print("DTW Distance:", dist)


def plot_alignment(sig1, sig2, path):
    sig1 = np.array(sig1)
    sig2 = np.array(sig2)

    plt.figure(figsize=(8, 6))
    plt.plot(sig1[:, 0], sig1[:, 1], 'bo-', label='Signature 1')
    plt.plot(sig2[:, 0], sig2[:, 1], 'ro-', label='Signature 2')

    for (i, j) in path:
        plt.plot([sig1[i][0], sig2[j][0]], [sig1[i][1], sig2[j][1]], 'k-', alpha=0.4)

    plt.legend()
    plt.title("DTW Alignment Between Two Signatures")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    # plt.show()

plot_alignment(sig1, sig2, path)
