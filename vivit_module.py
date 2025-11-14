import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
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


from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data      # shape: [N, T, F]
        self.labels = labels  # shape: [N]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


from torch.utils.data import DataLoader

model = TimeSeriesTransformer(input_dim=6)  # example: 6 features
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_loader = DataLoader(TimeSeriesDataset(train_X, train_y), batch_size=32, shuffle=True)

for epoch in range(20):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).squeeze() > 0.5
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total
