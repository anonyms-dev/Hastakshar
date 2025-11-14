import os
import numpy as np
from GMM_HMM_hindi import extract_features, load_signature_hmm
from dynamic_time_warping import *
# from vmd_decomposition import VMD, read_txt_signal
from emd_decomposition import
import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.signal import resample  # if you want to resample later
# from pyemd import emd


# this is to train only for chinese data 
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

import numpy as np
from scipy.signal import hilbert

def extract_hht_features(pressure_signal):
    """
    Extract HHT (Hilbert-Huang Transform) features from a 1D pressure signal.
    Returns a flat feature vector combining amplitude and frequency stats
    from all IMFs.
    """

    # Try importing EMD from PyEMD, else use fallback
    try:
        from PyEMD import EMD
        emd = EMD()
        IMFs = emd(pressure_signal)
    except ImportError:
        print(" PyEMD not found — using FFT-based pseudo decomposition instead.")
        # Fallback: split signal into frequency bands via FFT (like VMD-lite)
        fft_sig = np.fft.fft(pressure_signal)
        T = len(pressure_signal)
        freqs = np.fft.fftfreq(T)
        K = 4  # number of pseudo-IMFs
        bands = np.array_split(np.argsort(np.abs(freqs)), K)
        IMFs = []
        for b in bands:
            mask = np.zeros(T, dtype=complex)
            mask[b] = 1
            imf = np.fft.ifft(fft_sig * mask).real
            IMFs.append(imf)
        IMFs = np.array(IMFs)

    features = []
    for imf in IMFs:
        analytic_signal = hilbert(imf)
        amplitude = np.abs(analytic_signal)
        phase = np.unwrap(np.angle(analytic_signal))
        freq = np.diff(phase) / (2.0 * np.pi)
        if len(freq) == 0:
            continue
        # Collect mean/std stats
        features.append([
            np.mean(amplitude),
            np.std(amplitude),
            np.mean(freq),
            np.std(freq)
        ])

    if len(features) == 0:
        # Avoid empty case
        return np.zeros(4)
    return np.array(features).flatten()


# def extract_hht_features(pressure_signal):
#     emd1 = emd()
#     IMFs = emd(pressure_signal)
#     features = []
#     for imf in IMFs:
#         analytic_signal = hilbert(imf)
#         amplitude = np.abs(analytic_signal)
#         phase = np.unwrap(np.angle(analytic_signal))
#         freq = np.diff(phase) / (2.0 * np.pi)
#         # Statistical features
#         features.append([
#             np.mean(amplitude),
#             np.std(amplitude),
#             np.mean(freq),
#             np.std(freq)
#         ])
#     return np.array(features).flatten()
# -------------------------
# load_signature (safe alignment + debug prints)
# -------------------------
def load_signature(file_path):
    # load base columns (x,y,lastcol)
    with open(file_path) as f:
        n_cols = len(f.readline().split(' '))
    data = np.loadtxt(file_path, delimiter=None, usecols=(0, 1, n_cols - 1))
    print(data)
    # exit()

    # GMM-HMM features
    gmm_hmm_data = extract_features(data[:-1])
    gmm_hmm_data = np.asarray(gmm_hmm_data)
    if gmm_hmm_data.ndim == 1:
        gmm_hmm_data = gmm_hmm_data[:, np.newaxis]

    # VMD on last column
    t, signal = read_txt_signal(file_path)
    # signal = data[:, -1]
    # print(signal)
    modes, omega = VMD(signal=signal, alpha=2000, K=4, N_iter=500)
    modes = np.transpose(modes)
    # modes = np.asarray(modes)
    # imf_signal = extract_hht_features(signal)
    # imf_signal= emd(signal, max_imfs=5)
    # modes shape depends on VMD impl; ensure modes is (T, K)
    # if modes.ndim == 2 and modes.shape[0] < modes.shape[1]:
    #     modes = modes.T

    # Debug shapes
    # print(f"data: {data.shape}, gmm_hmm: {gmm_hmm_data.shape}, modes: {modes.shape}")
    # exit()

    # Align all to same time length by trimming to min (safe)
    min_len = min(len(data), len(gmm_hmm_data), len(modes))
    data = data[:min_len]
    gmm_hmm_data = gmm_hmm_data[:min_len]
    modes = modes[:min_len]
    print(np.shape(data), np.shape(gmm_hmm_data), np.shape(modes))
    # Concatenate features along feature axis
    temp_data = np.concatenate((data, gmm_hmm_data, modes), axis=1)  # shape (T, F_total)
    print("temp", np.shape(temp_data))

    # exit()
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
    data_root = "/Users/balaji.raok/Documents/online_signature_work/sigComp2011_Full_Dataset/sigComp2011-trainingSet/OnlineSignatures/Chinese/TrainingSet"
    samples, labels = [], []
    print(os.listdir(data_root))
    # exit()
    for user in os.listdir(data_root):
        if user.startswith('.') and user != '.':
            continue
        user_path = os.path.join(data_root, user)
        for fname in os.listdir(user_path):
            fpath = os.path.join(user_path, fname)

    # Skip hidden files and directories
        if fname.startswith('.') or os.path.isdir(fpath):
            print(f"Skipping: {fpath}")
            continue

        try:
            print(fpath)
            exit()
            data = load_signature(fpath)
            samples.append(data)
            labels.append(0 if 'F' in user_path else 1)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            continue

        # Label assignment
        # if 'F' in fname:
        #     labels.append(0)
        # else:
        #     labels.append(1)
        #     seq = load_signature(user_path)  # nd-array (T, F)
        
        # continue
    # samples.append(seq)
    

    print(f"Loaded {len(samples)} samples.")
    print("samples", samples)
    # exit()

    # Train/val split
    train_X, val_X, train_y, val_y = train_test_split(samples, labels, test_size=0.2, random_state=16)

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

def main():
    data_root = "/Users/balaji.raok/Documents/online_signature_work/sigComp2011_Full_Dataset/sigComp2011-trainingSet/OnlineSignatures/Chinese/TrainingSet/"
    
    genuine_path = os.path.join(data_root, "Online_Genuine")
    forge_path = os.path.join(data_root, "Online_Forgeries")

    samples, labels = [], []

    # --------------------
    # Load Genuine samples
    # --------------------
    for fname in os.listdir(genuine_path):
        if fname.startswith('.') or os.path.isdir(os.path.join(genuine_path, fname)):
            continue
        fpath = os.path.join(genuine_path, fname)
        print(f"Loading Genuine: {fpath}")
        # exit()
        # try:
        print(fpath)
        # exit()
        data = load_signature(fpath)
        print(np.shape(data))
        # exit()
        samples.append(data)
        labels.append(1)   # 1 = Genuine
        # except Exception as e:
        #     print(f"Error loading {fpath}: {e}")
        #     continue

    # --------------------
    # Load Forged samples
    # --------------------
    for fname in os.listdir(forge_path):
        if fname.startswith('.') or os.path.isdir(os.path.join(forge_path, fname)):
            continue
        fpath = os.path.join(forge_path, fname)
        print(f"Loading Forged: {fpath}")
        # try:
        data = load_signature(fpath)
        samples.append(data)
        labels.append(0)   # 0 = Forged
        # except Exception as e:
        #     print(f"Error loading {fpath}: {e}")
        #     continue

    print(f"Loaded {len(samples)} samples. Genuine={labels.count(1)}, Forged={labels.count(0)}")

    # --------------------
    # Train/val split
    # --------------------
    # if not samples:
    #     raise RuntimeError("No valid samples loaded!")

    feature_dim = samples[0].shape[1]
    # print(f"Feature dim: {feature_dim}")
    # print(samples, labels)
    # exit()

    train_X, val_X, train_y, val_y = train_test_split(samples, labels, test_size=0.2, random_state=16)

    train_loader = DataLoader(HindiOnlineDataset(train_X, train_y), batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(HindiOnlineDataset(val_X, val_y), batch_size=16, shuffle=False, collate_fn=collate_fn)

    # --------------------
    # Model + Training Loop
    # --------------------
    model = TemporalTransformer(input_dim=feature_dim).cpu()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y, mask, _ in train_loader:
            pred = model(x, mask).squeeze(1)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.6f}")

        # Validation
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
