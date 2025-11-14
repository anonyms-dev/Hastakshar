import os
import numpy as np
from GMM_HMM_hindi import extract_features, load_signature_hmm
from dynamic_time_warping import *
# from vmd_decomposition import VMD, read_txt_signal
from emd_decomposition import EMD_decomp, read_txt_signal, extract_hht_features
import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler

# -------------------------
# Positional Encoding + Transformer
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=8000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
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
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        out = self.classifier(x)
        return out


# -------------------------
# load_signature
# -------------------------
def load_signature(file_path):
    print("path", file_path)
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()

    if ',' in first_line:
        delimiter = ','
        n_cols = len(first_line.split(','))
    elif '\t' in first_line:
        delimiter = '\t'
        n_cols = len(first_line.split('\t'))
    else:
        delimiter = None
        n_cols = len(first_line.split())

    data = np.loadtxt(file_path, delimiter=delimiter, usecols=(0, 1, -1))

    # GMM-HMM features
    gmm_hmm_data = extract_features(data[:-1])
    gmm_hmm_data = np.asarray(gmm_hmm_data)
    if gmm_hmm_data.ndim == 1:
        gmm_hmm_data = gmm_hmm_data[:, np.newaxis]

    signal = data[:, -1]
    modes, omega = EMD_decomp(signal=signal, alpha=2000, K=4, N_iter=500)
    modes = np.asarray(modes)
    if modes.ndim == 2 and modes.shape[0] < modes.shape[1]:
        modes = modes.T

    # Optional: HHT features
    try:
        hht_features = extract_hht_features(signal)
        hht_features = np.asarray(hht_features)
        if hht_features.ndim == 1:
            hht_features = hht_features[:, np.newaxis]
    except Exception:
        hht_features = np.zeros((len(signal), 1))

    min_len = min(len(data), len(gmm_hmm_data), len(modes), len(hht_features))
    data = data[:min_len]
    gmm_hmm_data = gmm_hmm_data[:min_len]
    modes = modes[:min_len]
    hht_features = hht_features[:min_len]

    temp_data = np.concatenate((data, gmm_hmm_data, modes, hht_features), axis=1)

    # Normalize
    scaler = StandardScaler()
    temp_data = scaler.fit_transform(temp_data)

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
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = [torch.as_tensor(s, dtype=torch.float32) for s in sequences]
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    max_len = lengths.max().item()
    padded_sequences = pad_sequence(sequences, batch_first=True)
    mask = torch.arange(max_len)[None, :].to(lengths.device) >= lengths[:, None]
    labels = torch.as_tensor(labels, dtype=torch.float32)
    return padded_sequences, labels, mask, lengths


# -------------------------
# Main training loop (with model saving)
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "temporal_transformer_signature.pt"

    data_root = "/Users/balaji.raok/Documents/online_signature_work/Complete_data_set/Genuine_set/"
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
                seq = load_signature(fpath)
            except Exception as e:
                print("Error loading", fpath, e)
                continue
            samples.append(seq)
            labels.append(0 if 'F' in fname else 1)

    print(f"Loaded {len(samples)} samples.")

    train_X, val_X, train_y, val_y = train_test_split(samples, labels, test_size=0.2, random_state=42)
    train_dataset = HindiOnlineDataset(train_X, train_y)
    val_dataset = HindiOnlineDataset(val_X, val_y)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    if len(train_X) == 0:
        raise RuntimeError("No training samples found.")
    feature_dim = train_X[0].shape[1]
    print("Feature dim:", feature_dim)

    model = TemporalTransformer(input_dim=feature_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_err = float("inf")

    # Optional: Resume training if checkpoint exists
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_val_err = checkpoint.get("best_val_err", float("inf"))
        print(f"✅ Loaded checkpoint from {save_path} with best val err {best_val_err:.4f}")

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for x, y, mask, lengths in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x, mask=mask).squeeze(1)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}")

        # Validation
        model.eval()
        total, incorrect = 0, 0
        with torch.no_grad():
            for x_val, y_val, mask_val, lengths_val in val_loader:
                x_val, y_val, mask_val = x_val.to(device), y_val.to(device), mask_val.to(device)
                preds = (model(x_val, mask=mask_val).squeeze(1) > 0.5).float()
                incorrect += (preds != y_val).sum().item()
                total += y_val.size(0)

        val_err = incorrect / total if total > 0 else 1.0
        print(f" Val Error Rate: {val_err * 100:.2f}%")

        # Save best model
        if val_err < best_val_err:
            best_val_err = val_err
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_err": best_val_err,
                "epoch": epoch + 1,
            }, save_path)
            print(f"✅ Saved best model at epoch {epoch+1} with val err {best_val_err:.4f}")

    print("Training complete. Best validation error:", best_val_err)


if __name__ == "__main__":
    main()
