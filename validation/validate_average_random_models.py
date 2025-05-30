"""
Trains the same neural network multiple times with fixed hyperparameters,
predicts a target value for the same input each time, and averages the predictions.
Useful for benchmarking, statistical validation, and checking if a single trained model
performs better than the average of randomly initialized models.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataclass import dataset

class NNmodel(nn.Module):
    def __init__(self, input_shape, n_layers, neurons):
        super().__init__()
        # Snap neurons to nearest power of 2 (optional, not strictly needed)
        neurons = [2 ** round((n).bit_length() - 1) for n in neurons]
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

def train_and_predict_average(path, year, month, temperature, slm_kgp, min_wage, target_value=213681.96, num_trials=10, num_epochs=600):
    # Load and preprocess dataset
    data = dataset(path)
    loader = DataLoader(data, batch_size=32, shuffle=True)

    # Model hyperparameters (static for all trials)
    input_shape = data.features.shape[1]
    n_layers = 6
    neurons = [64, 32, 64, 128, 32, 64]
    lr = 0.004444958200568065
    weight_decay = 0.0006210591925389088
    gamma = 0.884
    step_size = 31

    predictions = []

    for trial in range(num_trials):
        # Fresh model for each trial
        model = NNmodel(input_shape, n_layers, neurons)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(num_epochs):
            model.train()
            for batch in loader:
                inputs, targets = batch
                inputs, targets = inputs.float(), targets.float()
                targets = targets.view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Prepare features for prediction
        year_scaled = data.scaler.transform([[year]]).flatten()[0]
        month_encoded = data.le_months.transform([month.lower()])[0]
        quarter = (month_encoded // 3) + 1
        slm_kgp_scaled = data.scaler_kg.transform([[slm_kgp]]).flatten()[0]
        temperature_scaled = data.scaler_temp.transform([[temperature]]).flatten()[0]
        min_wage_scaled = data.scaler_minwages.transform([[min_wage]]).flatten()[0]

        input_features = np.array([
            year_scaled, month_encoded, temperature_scaled, quarter, slm_kgp_scaled, min_wage_scaled
        ], dtype=np.float64)
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)

        # Predict and inverse transform
        model.eval()
        with torch.no_grad():
            prediction_scaled = model(input_tensor).item()
        prediction = data.target_scaler.inverse_transform([[prediction_scaled]]).flatten()[0]
        predictions.append(prediction)

        print(f"Trial {trial + 1}: Prediction: {prediction:.2f} | Δ: {abs(prediction - target_value):.2f}")

    avg_prediction = np.mean(predictions)
    print(f"\nAverage prediction over {num_trials} trials: {avg_prediction:.2f}")
    print(f"Average difference to target: {abs(avg_prediction - target_value):.2f}")

    return avg_prediction

# Example usage, replace with your values as needed
if __name__ == "__main__":
    path = "data/fake_dataset.xlsx"
    year = 2024
    month = 'october'
    temperature = 12.1
    slm_kgp = 1977
    min_wage = 11.44
    train_and_predict_average(path, year, month, temperature, slm_kgp, min_wage, num_trials=20)