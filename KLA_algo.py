import torch
import torch.nn as nn
from kan import KAN, create_dataset
import matplotlib.pyplot as plt

# Set default tensor type for high precision, as KANs can benefit from it
torch.set_default_dtype(torch.float64)

# --- 1. Create a Synthetic Dataset ---
# Define the target function f(x0, x1) = exp(sin(pi*x0) + x1^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, train_num=1000, test_num=1000)
# print("dataset", dataset)
# exit()
# dataset is a dictionary with keys: 'train_input', 'train_label', 'test_input', 'test_label'
X_train = dataset['train_input']
y_train = dataset['train_label']
print(X_train)
# print( y_train)
# exit()

# --- 2. Define the KAN Model Architecture ---
# The architecture is defined by the number of neurons in each layer.
# [2, 5, 1] means:
# - Input layer with 2 features (since n_var=2)
# - One hidden layer with 5 neurons
# - Output layer with 1 neuron
model = KAN(width=[2, 5, 1], grid=5, k=3, seed=0)
# 'grid' controls the number of grid points for the B-splines
# 'k' is the order of the B-splines

# --- 3. Training the KAN Model ---
# You can use the built-in training function for convenience
# Note: Training KANs often involves a two-stage process: pre-training and symbolic-regression-friendly training
results = model.fit(dataset, opt="LBFGS", steps=50)
# results = model.train(dataset) 
print(results)
# opt="LBFGS" is often preferred for KANs for its fast convergence in this context
# 'steps' is the number of LBFGS optimization steps

# --- 4. Evaluate (Optional) ---
train_loss = results['train_loss'][-1]
# test_loss = model.eval(dataset)['test_loss']
print(f"Final Train Loss: {train_loss:.6f}")
# print(f"Final Test Loss: {test_loss:.6f}")

# --- 5. Visualization for Interpretability ---
# KANs are highly interpretable. This command visualizes the network.

model.plot(beta=100, scale=0.8)
plt.show()