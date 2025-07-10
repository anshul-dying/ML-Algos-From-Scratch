import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

class LinearDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def train_model(model, dataloader, criterion, optimizer, epochs=100):
    loss_history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return model, loss_history



def plot_loss_curve(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()


def plot_predictions(model, X, y):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        predictions = model(X_tensor).numpy()

    plt.figure()
    plt.scatter(y, predictions, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.title("Predicted vs True")
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.grid(True)
    plt.show()


def main():
    np.random.seed(0)
    X = np.random.rand(1000, 3)
    true_weights = np.array([[2.0], [-1.0], [0.5]])
    y = X @ true_weights + 1.5 + 0.1 * np.random.randn(1000, 1)

    dataset = LinearDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LinearRegressionModel(input_dim=3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model, loss_history = train_model(model, dataloader, criterion, optimizer, epochs=100)

    print("Model training complete and saved.")

    plot_loss_curve(loss_history)
    plot_predictions(model, X, y)


if __name__ == '__main__':
    main()
