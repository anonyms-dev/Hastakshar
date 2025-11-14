import os
import numpy as np
from GMM_HMM_hindi import extract_features, load_signature_hmm
from dynamic_time_warping import *
from vmd_decomposition import VMD, read_txt_signal
import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.signal import resample  # if you want to resample later

# -------------------------
# Positional Encoding + Transformer
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        """
        x: [B, T, F]
        mask: [B, T] where True = padding positions (src_key_padding_mask)
        """
        x = self.input_proj(x)             # [B, T, d_model]
        x = self.pos_encoder(x)            # add positional encoding
        # PyTorch Transformer expects mask where True==PAD
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, T, d_model]
        # Aggregate only real positions — since padding is masked during attention,
        # mean over time still ok (but could use lengths-aware pooling)
        x = x.mean(dim=1)                  # [B, d_model]
        out = self.classifier(x)           # [B, 1]
        return out


# -------------------------
# load_signature (safe alignment + debug prints)
# -------------------------
def load_signature(file_path):
    # load base columns (x,y,lastcol)
    with open(file_path) as f:
        n_cols = len(f.readline().split(','))
    data = np.loadtxt(file_path, delimiter=',', usecols=(0, 1, n_cols - 1))

    # GMM-HMM features
    gmm_hmm_data = extract_features(data[:-1])
    gmm_hmm_data = np.asarray(gmm_hmm_data)
    if gmm_hmm_data.ndim == 1:
        gmm_hmm_data = gmm_hmm_data[:, np.newaxis]

    # VMD on last column
    t, signal = read_txt_signal(file_path)
    signal = data[:, -1]
    modes, omega = VMD(signal=signal, alpha=2000, K=4, N_iter=500)
    modes = np.asarray(modes)
    # modes shape depends on VMD impl; ensure modes is (T, K)
    if modes.ndim == 2 and modes.shape[0] < modes.shape[1]:
        modes = modes.T

    # Debug shapes
    # print(f"data: {data.shape}, gmm_hmm: {gmm_hmm_data.shape}, modes: {modes.shape}")

    # Align all to same time length by trimming to min (safe)
    min_len = min(len(data), len(gmm_hmm_data), len(modes))
    data = data[:min_len]
    gmm_hmm_data = gmm_hmm_data[:min_len]
    modes = modes[:min_len]

    # Concatenate features along feature axis
    temp_data = np.concatenate((data, gmm_hmm_data, modes), axis=1)  # shape (T, F_total)

    return temp_data


# -------------------------
# Dataset / collate
# -------------------------
class HindiOnlineDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # return numpy arrays as-is; collate_fn will convert to tensors
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    """
    batch: list of (sequence (np or tensor), label)
    returns: padded_sequences [B, T_max, F], labels [B], mask [B, T_max], lengths [B]
    """
    sequences, labels = zip(*batch)
    # convert to tensors
    sequences = [torch.as_tensor(s, dtype=torch.float32) for s in sequences]

    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    max_len = lengths.max().item()

    # pad
    padded_sequences = pad_sequence(sequences, batch_first=True)  # [B, T_max, F]

    # mask True where padding
    mask = torch.arange(max_len)[None, :].to(lengths.device) >= lengths[:, None]

    labels = torch.as_tensor(labels, dtype=torch.float32)
    return padded_sequences, labels, mask, lengths


# -------------------------
# Main training loop (fixed)
# -------------------------
def main():
    # Load data files
    data_root = "/Users/balaji.raok/Documents/online_signature_work/Hindi_sign_db/Subset/genuine_set/"
    samples, labels = [], []
    for user in os.listdir(data_root):
        if user.startswith('.') and user != '.':
            continue
        user_path = os.path.join(data_root, user)
        if not os.path.isdir(user_path):
            continue
        for fname in os.listdir(user_path):
            if fname.startswith('.') and fname != '.':
                continue
            fpath = os.path.join(user_path, fname)
            try:
                seq = load_signature(fpath)  # nd-array (T, F)
            except Exception as e:
                print("Error loading", fpath, e)
                continue
            samples.append(seq)
            labels.append(0 if 'F' in fname else 1)

    print(f"Loaded {len(samples)} samples.")

    # Train/val split
    train_X, val_X, train_y, val_y = train_test_split(samples, labels, test_size=0.2, random_state=42)

    train_dataset = HindiOnlineDataset(train_X, train_y)
    val_dataset = HindiOnlineDataset(val_X, val_y)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # infer input_dim from first sample's feature dimension
    if len(train_X) == 0:
        raise RuntimeError("No training samples found.")
    feature_dim = train_X[0].shape[1]
    print("Feature dim:", feature_dim)

    model = TemporalTransformer(input_dim=feature_dim).cpu()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y, mask, lengths in train_loader:   # <-- unpack correctly!
            # x: [B, T, F], y: [B], mask: [B, T]
            x = x.cpu()
            y = y.cpu()
            mask = mask.cpu()

            pred = model(x, mask=mask).squeeze(1)  # [B]
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch+1}: Loss = {total_loss / max(1, n_batches):.6f}")

        # (optional) validation step
        model.eval()
        with torch.no_grad():
            total, incorrect = 0, 0
            for x_val, y_val, mask_val, lengths_val in val_loader:
                x_val = x_val.cpu()
                y_val = y_val.cpu()
                mask_val = mask_val.cpu()
                preds = (model(x_val, mask=mask_val).squeeze(1) > 0.5).float()
                incorrect += (preds != y_val).sum().item()
                total += y_val.size(0)
            if total > 0:
                err = incorrect / total
                print(f" Val Error Rate: {err * 100:.2f}%")

if __name__ == "__main__":
    main()
