import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
from utils.excel_bot import ExcelBot

class dataset(Dataset):
    """
    Torch dataset for loading and preprocessing Excel data for regression tasks.
    - Encodes months, computes quarters, and scales all features and targets.
    - Returns (features, target) pairs for PyTorch DataLoader.
    """
    def __init__(self, file_path):
        # Load data from Excel file
        self.data = []
        excel = ExcelBot(file_path)
        for row in excel.sheet.iter_rows(min_row=2, max_row=excel.sheet.max_row, min_col=1, max_col=7):
            row_data = [cell.value for cell in row]
            self.data.append(row_data)

        # Encode months
        months = [row[1].lower() for row in self.data]
        self.le_months = LabelEncoder()
        encoded_months = self.le_months.fit_transform(months)

        # Calculate quarters (optional feature)
        quarters = [(month // 3) + 1 for month in encoded_months]

        # Scale features
        years = [[row[0]] for row in self.data]
        self.scaler = StandardScaler()
        scaled_years = self.scaler.fit_transform(years).flatten()

        slm_kgp = [[row[2]] for row in self.data]
        self.scaler_kg = StandardScaler()
        scaled_slm_kgps = self.scaler_kg.fit_transform(slm_kgp).flatten()

        temperature = [[row[3]] for row in self.data]
        self.scaler_temp = StandardScaler()
        scaled_temperatures = self.scaler_temp.fit_transform(temperature).flatten()

        min_wages = [[row[4]] for row in self.data]
        self.scaler_minwages = StandardScaler()
        scaled_min_wages = self.scaler_minwages.fit_transform(min_wages).flatten()

        # Build features and targets
        self.features = []
        self.targets = []
        for i, row in enumerate(self.data):
            scaled_year = scaled_years[i]
            scaled_slm_kgp = scaled_slm_kgps[i]
            scaled_temperature = scaled_temperatures[i]
            scaled_min_wage = scaled_min_wages[i]
            quarter = quarters[i]
            self.features.append([scaled_year, encoded_months[i], scaled_temperature, quarter, scaled_slm_kgp, scaled_min_wage])
            self.targets.append(row[5])

        # Scale targets
        self.target_scaler = StandardScaler()
        self.targets = self.target_scaler.fit_transform(np.array(self.targets).reshape(-1, 1)).flatten()

        self.features = np.array(self.features, dtype=np.float64)
        self.targets = np.array(self.targets, dtype=np.float64)

    # Number of samples
    def __len__(self):
        return len(self.features)

    # Return (features, target) pair
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
