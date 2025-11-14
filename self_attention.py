import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from GMM_HMM_hindi import extract_features, load_signature_hmm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# from time_series_transformer import collate_fn
from torch.nn.utils.rnn import pad_sequence
import seaborn as sns

class SignatureSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, x):
        # x: [T, D]
        Q = self.query(x)       # [T, D]
        K = self.key(x)         # [T, D]
        V = self.value(x)       # [T, D]

        scores = torch.matmul(Q, K.T) * self.scale  # [T, T]
        weights = F.softmax(scores, dim=-1)         # [T, T]
        output = torch.matmul(weights, V)           # [T, D]
        return output, weights


T, D = 50, 6  # 50 time steps, 6 features
signature = torch.rand(T, D)


from torch.utils.data import Dataset

class HindiOnlineDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_signature(file_path):
    print(file_path)
    data = np.loadtxt(file_path, delimiter=',', usecols=(0, 1))
    load_raw = load_signature_hmm(file_path)
    print(np.shape(data[0:-1]))
    
    # print("lll", load_raw)
    # exit()
    # test_feat = extract_features(test_raw)
    # print("data", data)
    gmm_hmm_data= extract_features(load_raw)
    print(np.shape(gmm_hmm_data))
    temp_data = np.concatenate((data[0:-1], gmm_hmm_data), axis= 1)
    print(temp_data)
    # exit()

    if data.ndim == 1:  # in case of single line
        data = data[np.newaxis, :]
    return temp_data#data[:, :2]  # Use x, y, only


def load_hindi_dataset_online(data_dir):
    samples, labels = [], []
    # print(os.listdir(data_dir))


    for user in os.listdir(data_dir):
        print(user)
        if user.startswith('.') and user != '.': # Skips .DS_Store, .git, etc.
            print(f"Skipping hidden file/folder: {user}")
            continue
        else:

            user_path = os.path.join(data_dir, user)

            
            # print(os.listdir(user_path))
            for fname in os.listdir(user_path):
                if fname.startswith('.') and fname != '.': # Skips .DS_Store, .git, etc.
                    print(f"Skipping hidden file/folder: {user}")
                    continue
                # print(user_path)
                # print(fname == '.DS_Store')
                # print(os.path.exists(os.path.join(user_path, fname)))
                

                try:

                    fpath = os.path.join(user_path, fname)
                    # print(fpath)

                    data = load_signature(fpath)
                    samples.append(data)
            
                except FileNotFoundError:

                    print(f"Directory '{fpath}' does not exist.")
                    continue

                if 'F' in fname:
                    labels.append(0)
                else:
                    labels.append(1)
    
    return samples, labels


def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in sequences],
                          batch_first=True)  # [B, T, F]
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, labels


samples, labels = load_hindi_dataset_online("/Users/balaji.raok/Documents/online_signature_work/Hindi_sign_db/Genuine_subset/")
print(np.shape(samples[0]), labels)
signature = torch.tensor(samples[2])
# exit()
# samples = torch.from_numpy(samples[0])
train_X, val_X, train_y, val_y = train_test_split(samples, labels, test_size=0.2, random_state=42)

train_dataset = HindiOnlineDataset(train_X, train_y)
val_dataset = HindiOnlineDataset(val_X, val_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# print(np.shape(train_dataset[0]))

D=6
attn_model = SignatureSelfAttention(d_model=D)
# print(train_dataset[0].shape)
output, attn_weights = attn_model(signature)  # attn_weights: [T, T]

import matplotlib.pyplot as plt

# def visualize_attention(attn_weights, title="Self-Attention Heatmap"):
#     T = attn_weights.size(0)
#     fig, ax = plt.subplots(figsize=(6, 5))
#     im = ax.imshow(attn_weights.detach().cpu(), cmap="viridis", aspect='auto')

#     ax.set_title(title)
#     ax.set_xlabel("Attended Time Step")
#     ax.set_ylabel("Query Time Step")
#     fig.colorbar(im, ax=ax, label='Attention Weight')
#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt

def visualize_attention(attn_weights, title="Self-Attention Heatmap"):
    # Convert torch tensor to a plain Python list (avoids NumPy)
    attn_list = attn_weights.detach().cpu().tolist()
    # plt.figure(figsize=(6, 5))

    # T = len(attn_list)
    # sns.heatmap(attn_list, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels,
    #         cbar_kws={'label': 'Attention Weight'}, square=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(attn_list, cmap="viridis", aspect='auto')

    ax.set_title(title)
    ax.set_xlabel("Attended Time Step")
    ax.set_ylabel("Query Time Step")
    fig.colorbar(im, ax=ax, label='Attention Weight')

    plt.tight_layout()
    plt.show()
    # plt.figure(figsize=(6, 5))
# sns.heatmap(attn, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels,
#             cbar_kws={'label': 'Attention Weight'}, square=True)

plt.title('Self-Attention Between Signatures')
plt.xlabel('Attended To')
plt.ylabel('Query')
plt.tight_layout()
plt.show()


# Call the visualizer
visualize_attention(attn_weights)


