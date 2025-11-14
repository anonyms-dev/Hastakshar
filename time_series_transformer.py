import os
import numpy as np
from GMM_HMM_hindi import extract_features, load_signature_hmm
from dynamic_time_warping import *
from time_series_transformer import collate_fn

import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_proj(x)         # [B, T, D]
        x = self.pos_encoder(x)        # Add positional encoding
        x = self.transformer(x)        # [B, T, D]
        x = x.mean(dim=1)              # Global average pooling
        return self.classifier(x)

import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=3500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


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


import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in sequences],
                          batch_first=True)  # [B, T, F]
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, labels


from torch.utils.data import Dataset

class HindiOnlineDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Load data
samples, labels = load_hindi_dataset_online("/Users/balaji.raok/Documents/online_signature_work/Hindi_sign_db/Genuine_subset/")
print(len(samples), labels)
# exit()
train_X, val_X, train_y, val_y = train_test_split(samples, labels, test_size=0.2, random_state=42)

train_dataset = HindiOnlineDataset(train_X, train_y)
val_dataset = HindiOnlineDataset(val_X, val_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
print(train_loader)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

# Model
model = TimeSeriesTransformer(input_dim=6).cpu()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.cpu(), y.cpu()
        pred = model(x).squeeze()
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")


# def evaluate(model, dataloader):
#     model.eval()
#     total, correct = 0, 0
#     with torch.no_grad():
#         for x, y in dataloader:
#             x, y = x.cpu(), y.cpu()
#             pred = (model(x).squeeze() > 0.5).float()
#             correct += (pred == y).sum().item()
#             total += y.size(0)
#     return correct / total

# acc = evaluate(model, val_loader)
# print(f"Validation Accuracy: {acc * 100:.2f}%")

def evaluate_error_rate(model, dataloader):
    model.eval()
    total, incorrect = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.cpu(), y.cpu()
            pred = (model(x).squeeze() > 0.5).float()
            incorrect += (pred != y).sum().item()
            total += y.size(0)
    error_rate = incorrect / total
    return error_rate


err = evaluate_error_rate(model, val_loader)
print(f"Validation Error Rate: {err * 1:.2f}%")



import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, T, F]
        x = self.input_proj(x)             # [B, T, d_model]
        x = self.pos_encoder(x)            # add positional encoding
        x = self.transformer(x)            # [B, T, d_model]
        x = x.mean(dim=1)                  # global average pooling
        out = self.classifier(x)           # [B, 1]
        return out


import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # [B, T, D]
        return self.dropout(x)


if __name__ == "__main__":
    main()