import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataclass import dataset

# NN architecture
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

# Manual test set
features_test = [
    [2024, "January", 1938, 5, 11.44, 322877.25],
    [2024, "February", 1966, 7.60, 11.44, 285816.51],
    [2024, "March", 1887, 7.90, 11.44, 269496.92],
    [2024, "April", 1960, 9.50, 11.44, 309541.88],
    [2024, "May", 1971, 13.80, 11.44, 304560.87],
    [2024, "June", 1985, 14.30, 11.44, 323528.57],
    [2024, "July", 1900, 16.20, 11.44, 271291.93],
    [2024, "August", 2006, 16.80, 11.44, 306547.02],
    [2024, "September", 1896, 13.80, 11.44, 288488.74],
    [2024, "October", 2070, 10.6, 11.44, 312001.35],
    [2024, "November", 1977, 6.6, 11.44, 311773.64],
    [2024, "December", 2055, 6.2, 11.44, 337604.29],
]

best_model_path = "models/best_model.pth"
data_path = "data/fake_dataset.xlsx"
data = dataset(data_path)
checkpoint = torch.load(best_model_path)

# --- AUTODETECT input_shape & neurons ---
params = checkpoint.get("params", None)
if params is not None:
    neurons = params["neurons"]
else:
    neurons = [32, 128, 64, 64, 64, 64]
input_shape = data.features.shape[1]
n_layers = len(neurons)

model = NNmodel(input_shape, n_layers, neurons)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Evaluate best model
test_differs = []
for features in features_test:
    year_scaled = data.scaler.transform([[features[0]]]).flatten()[0]
    month_encoded = data.le_months.transform([features[1].lower()])[0]
    quarter = (month_encoded // 3) + 1
    slm_kgp_scaled = data.scaler_kg.transform([[features[2]]]).flatten()[0]
    temperature_scaled = data.scaler_temp.transform([[features[3]]]).flatten()[0]
    min_wage_scaled = data.scaler_minwages.transform([[features[4]]]).flatten()[0]
    input_features = np.array([
        year_scaled, month_encoded, temperature_scaled, quarter, slm_kgp_scaled, min_wage_scaled
    ], dtype=np.float64)
    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
    target_val = features[5]
    with torch.no_grad():
        prediction_scaled = model(input_tensor).item()
    prediction = data.target_scaler.inverse_transform([[prediction_scaled]]).flatten()[0]
    test_differs.append(abs(prediction - target_val))
best_model_avg = sum(test_differs) / len(test_differs)

# Retrain/test several random models to compare
average_better_count = 0
results = []

for repeat in range(10):
    print(f"Repeat {repeat + 1}/10")
    averages = []
    for trial in range(20):
        full_dataset = DataLoader(data, batch_size=32, shuffle=True)
        trial_model = NNmodel(input_shape, n_layers, neurons)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(trial_model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=31, gamma=0.884)
        for epoch in range(500):
            trial_model.train()
            for batch in full_dataset:
                inputs, targets = batch
                inputs, targets = inputs.float(), targets.float()
                targets = targets.view(-1, 1)
                optimizer.zero_grad()
                outputs = trial_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()
        # Validate on the manual set
        test_differs = []
        for features in features_test:
            year_scaled = data.scaler.transform([[features[0]]]).flatten()[0]
            month_encoded = data.le_months.transform([features[1].lower()])[0]
            quarter = (month_encoded // 3) + 1
            slm_kgp_scaled = data.scaler_kg.transform([[features[2]]]).flatten()[0]
            temperature_scaled = data.scaler_temp.transform([[features[3]]]).flatten()[0]
            min_wage_scaled = data.scaler_minwages.transform([[features[4]]]).flatten()[0]
            input_features = np.array([
                year_scaled, month_encoded, temperature_scaled, quarter, slm_kgp_scaled, min_wage_scaled
            ], dtype=np.float64)
            input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
            target_val = features[5]
            trial_model.eval()
            with torch.no_grad():
                prediction_scaled = trial_model(input_tensor).item()
            prediction = data.target_scaler.inverse_transform([[prediction_scaled]]).flatten()[0]
            test_differs.append(abs(prediction - target_val))
        local_average = sum(test_differs) / len(test_differs)
        averages.append(local_average)
    avg_avg = sum(averages) / len(averages)
    if avg_avg > best_model_avg:
        difference = avg_avg - best_model_avg
        results.append(f"Repeat {repeat + 1}: Best model performed better by {difference:.4f}")
    else:
        difference = best_model_avg - avg_avg
        average_better_count += 1
        results.append(f"Repeat {repeat + 1}: Average performed better by {difference:.4f}")

print("\nSummary of Results:")
for result in results:
    print(result)
print(f"\nThe average performed better in {average_better_count} out of 10 repetitions.")