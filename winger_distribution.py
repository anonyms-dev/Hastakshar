import torch
from pykan.kan import KAN  # or adjust import path per version

# Define toy data
X = torch.linspace(-2, 2, 500).unsqueeze(1)
Y = torch.sin(X * 2.0) + 0.3 * torch.randn_like(X)

# Initialize model
model = KAN(width=[1, 10, 1], grid=3, k=3)  # example hyper‐parameters

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(2000):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.6f}")
