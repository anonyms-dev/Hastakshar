import torch
from pykan import KAN

# Create a simple dataset
x = torch.linspace(-3, 3, 500).unsqueeze(1)
y = torch.sin(2 * x) + 0.3 * torch.randn_like(x)

# Define Kolmogorov-Arnold Network
model = KAN(width=[1, 8, 1])  # similar to MLP structure

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
