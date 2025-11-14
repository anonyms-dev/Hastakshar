import numpy as np
import os
from glob import glob
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def load_signature_file(filepath):
    print(".......", filepath)
    data = np.loadtxt(filepath, delimiter=',', usecols=(0, 1))
    print("data", data)
    return data[:, :2]
    # Keep only x, y, pressure (if available)
    # if data.shape[1] >= 3:
    #     print(data)
    #     exit()
    #     return data[:, :3]
    #     # print(data)
    # else:
    #     return data[:, :2]  # fallback to (x, y)
    


def normalize_signature(sig):
    return (sig - np.mean(sig, axis=0)) / (np.std(sig, axis=0) + 1e-8)


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(s1[i - 1], s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                   dtw[i, j - 1],
                                   dtw[i - 1, j - 1])
            
    
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        directions = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
        i, j = min(directions, key=lambda x: dtw[x[0], x[1]])
    path.reverse()
    print(path[0])

    return dtw[n, m], path

def plot_signature_alignment(sig1, sig2):
    plt.plot(sig1[:, 0], sig1[:, 1], 'b-', label='Genuine')
    plt.plot(sig2[:, 0], sig2[:, 1], 'r--', label='Test')
    plt.legend()
    plt.title("Signature Alignment")
    plt.gca().invert_yaxis()  # Stylus Y is often inverted
    plt.show()

# def plot_alignment(sig1, sig2, path):
#     sig1 = np.array(sig1)
#     sig2 = np.array(sig2)

#     plt.figure(figsize=(8, 6))
#     plt.plot(sig1[:, 0], sig1[:, 1], 'bo-', label='Signature 1')
#     plt.plot(sig2[:, 0], sig2[:, 1], 'ro-', label='Signature 2')

#     for (i, j) in path:
#         plt.plot([sig1[i][0], sig2[j][0]], [sig1[i][1], sig2[j][1]], 'k-', alpha=0.4)

#     plt.legend()
#     plt.title("DTW Alignment Between Two Signatures")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.grid(True)
#     plt.show()

def plot_signature_alignment_with_path(sig1, sig2, path):
    sig1 = np.array(sig1)
    sig2 = np.array(sig2)

    plt.figure(figsize=(8, 6))
    plt.subplot(1, 3, 1)
    plt.plot(sig1[:, 0], sig1[:, 1], 'bo-', label='Genuine')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.title("Genuine Signature")
    plt.subplot(1, 3, 2)
    plt.plot(sig2[:, 0], sig2[:, 1], 'ro-', label='Forge')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.title("Forge Signature")


    plt.subplot(1, 3, 3)
    plt.plot(sig1[:, 0], sig1[:, 1], 'bo-', label='Genuine')
    plt.plot(sig2[:, 0], sig2[:, 1], 'ro-', label='Forge')

    # Plot warping lines
    for (i, j) in path:
        plt.plot([sig1[i][0], sig2[j][0]], [sig1[i][1], sig2[j][1]], 'k-', alpha=0.3)

    plt.legend()
    plt.title("DTW Alignment with Warping Path")
    plt.gca().invert_yaxis()  # for stylus-style coordinates
    plt.grid(True)
    plt.savefig("DTW_mapping_7.jpg")
    plt.show()


def evaluate_dtw_verification(genuine_dir, forgery_dir, threshold=15.0):
    genuine_files = sorted(glob(os.path.join(genuine_dir, "*.txt")))
    genuine_files = genuine_files[:1]
    # print(genuine_files[:5])
    # exit()
    forgery_files = sorted(glob(os.path.join(forgery_dir, "*.txt")))
    forgery_files = forgery_files[:1]
    # print(forgery_files[:5])
    # print(genuine_files[1])
    # data = load_signature_file(genuine_files[1])
    # print(data)
    # exit()
    # Use first 25 genuine as references
    references = [normalize_signature(load_signature_file(f)) for f in genuine_files[:1]]

    labels = []
    predictions = []

    # Test other genuine
    # for test_file in genuine_files[1:]:
    #     test = normalize_signature(load_signature_file(test_file))
    #     dists, path = dtw_distance(test, ref_sig)
    #     # dists, path = [dtw_distance(test, ref) for ref in references]
    #     mean_dist = np.mean(dists)
    #     labels.append(1)
    #     predictions.append(1 if mean_dist < threshold else 0)

    # # Test forgeries
    # for test_file in forgery_files:
    #     test = normalize_signature(load_signature_file(test_file))
    #     dists, path = dtw_distance(test, ref_sig)
    #     # dists, path= [dtw_distance(test, ref) for ref in references]
    #     mean_dist = np.mean(dists)
    #     labels.append(0)
    #     predictions.append(1 if mean_dist < threshold else 0)
    # print(ref_sig)
    # exit()
    ref_sig = normalize_signature(load_signature_file(genuine_files[0]))
    test_sig = normalize_signature(load_signature_file(forgery_files[0]))
    # print(ref_sig)
    _, path = dtw_distance(ref_sig, test_sig)
    plot_signature_alignment_with_path(ref_sig, test_sig, path=path)

    # plot_signature_alignment(ref_sig, test_sig)

    # acc = accuracy_score(labels, predictions)
    # print(f"Accuracy: {acc * 100:.2f}%")
    return labels, predictions


genuine_path = "/Users/balaji.raok/Documents/online_signature_work/Hindi_sign_db/Genuine_set/12_dec_2024/hassan"
forgery_path = "/Users/balaji.raok/Documents/online_signature_work/Hindi_sign_db/Forge_signature/30_dec_2024/Kabeer"

labels, preds = evaluate_dtw_verification(genuine_path, forgery_path, threshold=15.0)

if __name__ == "__main__":
    main()





