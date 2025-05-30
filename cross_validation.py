import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
from utils.dataclass import dataset
from sklearn.model_selection import KFold

# Base neural net
class NNmodel(nn.Module):
    def __init__(self, input_shape, n_layers, neurons):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_shape, neurons[0], bias=True))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(neurons[0]))
        for i in range(1, n_layers):
            self.layers.append(nn.Linear(neurons[i - 1], neurons[i], bias=True))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(neurons[i]))
        self.output = nn.Linear(neurons[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x

# Folder with all candidate models
models_folder = "models"
model_files = [f for f in os.listdir(models_folder) if f.endswith('.pth')]

# Data for validation
data_path = "data\\fake_dataset.xlsx"
data = dataset(data_path)
input_shape = data.features.shape[1]
k_folds = 5

# Track best model
best_mae = float('inf')
best_model_info = None

# Loop over models
for fname in model_files:
    checkpoint = torch.load(os.path.join(models_folder, fname))
    params = checkpoint['params']
    n_layers = len(params['neurons'])
    neurons = params['neurons']
    lr = params['lr']
    weight_decay = params['weight_decay']
    gamma = params['gamma']
    step_size = params['step_size']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        train_subset = Subset(data, train_idx)
        val_subset = Subset(data, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = NNmodel(input_shape, n_layers, neurons)
        model.load_state_dict(checkpoint['state_dict'])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.float(), targets.float()
                targets = targets.view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        true_vals, pred_vals = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.float(), targets.float()
                targets = targets.view(-1, 1)
                outputs = model(inputs)
                true_vals.extend(targets.cpu().numpy().flatten())
                pred_vals.extend(outputs.cpu().numpy().flatten())

        mae = np.mean(np.abs(np.array(true_vals) - np.array(pred_vals)))
        fold_maes.append(mae)

    mean_mae = np.mean(fold_maes)
    print(f"{fname}: CV MAE = {mean_mae:.4f}")

    if mean_mae < best_mae:
        best_mae = mean_mae
        best_model_info = {
            "state_dict": checkpoint["state_dict"],
            "params": params,
            "mae": mean_mae,
            "filename": fname
        }

# Save best model
if best_model_info:
    out_path = os.path.join(models_folder, "best_model.pth")
    torch.save(best_model_info, out_path)
    print(f"Best model saved to {out_path} with MAE {best_model_info['mae']:.4f}")
else:
    print("No model improved performance.")