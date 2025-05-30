import os
import torch
import optuna
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
from sklearn.model_selection import KFold
from utils.dataclass import dataset

path = "data\\fake_dataset.xlsx"
data = dataset(path)

# Create a base neural network with optuna tunnning
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
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d) and x.size(0) == 1:
                continue
            x = layer(x)
        x = self.output(x)
        return x

top_models = []

def objective(trial):
    n_layers = trial.suggest_int("n_layers", 5, 7) # Optuna tunes number of layers
    neurons = [trial.suggest_categorical(f"num_neurons_layer_{i}", [32, 64, 128]) for i in range(n_layers)] # Optuna tunes neurons
    lr = trial.suggest_float("lr", 0.0042, 0.0045, log=True) # Optuna tunes learning rate
    weight_decay = trial.suggest_float("weight_decay", 0.0006, 0.0007, log=True) # Optuna tunes weight decay
    gamma = trial.suggest_float("gamma", 0.80, 0.99)             # Optuna tunes gamma
    step_size = trial.suggest_int("step_size", 10, 60)           # Optuna tunes step_size
    batch_size = trial.suggest_categorical("batch_size", [32, 64])  # Optuna tunes batch_size
    num_epochs = trial.suggest_int("num_epochs", 80, 120)        # Optuna tunes number of epochs
    k_folds = 5
    input_shape = len(data.features[0])
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_maes = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(data)):
        train_subset = Subset(data, train_indices)
        val_subset = Subset(data, val_indices)
        train_data = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_data = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        model = NNmodel(input_shape=input_shape, n_layers=n_layers, neurons=neurons)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(num_epochs):
            model.train()
            for batch in train_data:
                inputs, targets = batch
                inputs, targets = inputs.float(), targets.float()
                targets = targets.view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        true_values = []
        predicted_values = []
        with torch.no_grad():
            for batch in val_data:
                inputs, targets = batch
                inputs, targets = inputs.float(), targets.float()
                targets = targets.view(-1, 1)
                outputs = model(inputs)
                true_values.extend(targets.cpu().numpy().flatten())
                predicted_values.extend(outputs.cpu().numpy().flatten())

        mae = np.mean(np.abs(np.array(true_values) - np.array(predicted_values)))
        fold_maes.append(mae)

    mean_mae = np.mean(fold_maes)
    top_models.append((mean_mae, model.state_dict(), trial.number, {
        'neurons': neurons,
        'lr': lr,
        'weight_decay': weight_decay,
        'gamma': gamma,
        'step_size': step_size,
        'batch_size': batch_size,
        'num_epochs': num_epochs
    }))
    top_models.sort(key=lambda x: x[0])
    if len(top_models) > 10:
        top_models.pop(-1)

    print(f"Trial {trial.number}: MAE {mean_mae:.4f}")
    return mean_mae

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

for i, (mae, state_dict, trial_num, params) in enumerate(top_models):
    fname = f"models/model_{i+1:02d}_mae_{mae:.4f}_trial_{trial_num}.pth"
    torch.save({'state_dict': state_dict, 'mae': mae, 'trial': trial_num, 'params': params}, fname)
print("Top 10 models saved in /models")

print("Best hyperparameters:", study.best_params)
print("Best MAE:", study.best_value)