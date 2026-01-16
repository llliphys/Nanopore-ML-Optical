import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 64), final_activation="ReLU"):
        super().__init__()

        layers = []
        layer_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(layer_dim, hidden_dim))
            layers.append(nn.ReLU())
            layer_dim = hidden_dim

        layers.append(nn.Linear(layer_dim, output_dim))  # no activation

        if final_activation == "ReLU":
            # Ensure non-negative output
            layers.append(nn.ReLU())
        if final_activation == "Sigmoid":
            # Force output to [0, 1] to match scaled Y
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def preprocess(X, Y):

    # --- Step 1: Preprocessing ---
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    # scaler_x = MinMaxScaler()
    # X_scaled = scaler_x.fit_transform(X)

    # # Check Y Skewness
    # is_skewed = check_skewness(Y)
    # if is_skewed:
    #     Y = np.log1p(Y)

    # Min-Max Scaling on Y
    scaler_y = MinMaxScaler()
    Y_scaled = scaler_y.fit_transform(Y)

    print(
        f"X_scaled Max: {X_scaled.max():.2f}, X_scaled min: {X_scaled.min():.2f}")
    print(
        f"Y_scaled Max: {Y_scaled.max():.2f}, Y_scale Min: {Y_scaled.min():.2f}")

    # Converting to PyTorch Tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

    # M, N = X_tensor.shape
    # _, L = Y_tensor.shape

    return X_tensor, Y_tensor


def check_skewness(Y, show_plot=False):

    is_skewed = False

    # Statistical check
    skewness = stats.skew(Y)
    print(f"Skewness of Y: {skewness}")

    # Visual check
    if show_plot:
        plt.hist(Y, bins=50)
        plt.title("Distribution of Y")
        plt.show()

    if skewness > 1:
        is_skewed = True

    return is_skewed


# Inverse Transformation (To get real values back)
def get_original_y(is_skewed, pred_scaled, scaler_y):

    # Step A: Reverse Min-Max
    original_y = scaler_y.inverse_transform(pred_scaled)
    # Step B: Reverse Log1p (Exponential - 1)
    if is_skewed:
        original_y = np.expm1(original_y)

    return original_y


def predict(model, X_tensor):

    model.eval()
    Y_pred_tensor = model(X_tensor)
    Y_pred_scaled = Y_pred_tensor.detach().numpy()

    return Y_pred_scaled


# General parameters
n_samples = 10000
n_features = 100
n_targets = 2
epochs = 1000
lr_rate = 1e-3
hidden_dims = (128, 64, 32)
final_activation = "ReLU"

# Make datasets X/Y for training and testing
X, Y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_targets=n_targets,
    noise=0,
    random_state=123
)

# Making Y large-scale and non-negative
Y = Y.reshape(-1, n_targets)
Y = np.abs(Y * 1e5) + 1e-6

print(f"X shape: {X.shape}, Y shape: {Y.shape}")
print(f"Y Max: {Y.max():.2f}, Y Min: {Y.min():.2f}")


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

X_tensor, Y_tensor = preprocess(X_train, Y_train)

model = MLP(input_dim=X_tensor.shape[1],
            output_dim=Y_tensor.shape[1], hidden_dims=hidden_dims)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr_rate)


# --- Training Loop ---
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


X_tensor, Y_tensor = preprocess(X_test, Y_test)
Y_pred_scaled = predict(model, X_tensor)

scaler_y = MinMaxScaler().fit(Y_test)
Y_pred = scaler_y.inverse_transform(Y_pred_scaled)


R2 = np.sqrt(r2_score(Y_pred, Y_test))
RMSE = np.sqrt(mean_squared_error(Y_pred, Y_test))

print(f"R2 = {R2:.3f}")
print(f"RMSE = {RMSE:.3f}")

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.plot(Y_test, Y_test, "k", lw=3)
ax.plot(Y_test, Y_pred, "bo", ms=10, alpha=0.5)

plt.show()

# X_test, Y_test = make_regression(
#     n_samples=100,
#     n_features=100,
#     n_targets=n_targets,
#     noise=0.5,
#     random_state=123
# )

# # Making Y large-scale and non-negative
# Y_test = Y_test.reshape(-1, n_targets)
# Y_test = np.abs(Y_test * 1e5)

# print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
# print(f"Y_test Max: {Y_test.max():.2f}, Y_test Min: {Y_test.min():.2f}")

# # --- Step 1: Preprocessing ---
# scaler_x = StandardScaler()
# X_test_scaled = scaler_x.fit_transform(X_test)

# # Min-Max Scaling on Y
# scaler_y = MinMaxScaler()
# Y_test_scaled = scaler_y.fit_transform(Y_test)

# # Converting to PyTorch Tensors
# X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
# Y_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)

# def predict(model, scaler_x, scaler_y, X_new_np, use_log=False):
#     # 1. Ensure the model is in evaluation mode (turns off dropout/batchnorm)
#     model.eval()

#     with torch.no_grad():  # Disable gradient calculation for efficiency
#         # 2. Scale the new input using the training scaler
#         X_scaled = scaler_x.transform(X_new_np)
#         X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

#         # 3. Pass through the model
#         predictions_scaled = model(X_tensor)

#         # 4. Convert back to Numpy
#         predictions_np = predictions_scaled.numpy()

#         # 5. Reverse the Min-Max scaling
#         y_inv_minmax = scaler_y.inverse_transform(predictions_np)

#         # 6. Reverse the Log1p (if you used it)
#         if use_log:
#             final_predictions = np.expm1(y_inv_minmax)
#         else:
#             final_predictions = y_inv_minmax

#     return final_predictions
