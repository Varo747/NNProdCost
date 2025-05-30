import pandas as pd
import numpy as np

file_path = 'data\\data_reference.xlsx'
df = pd.read_excel(file_path)

# Strip spaces for safety
df.columns = [col.strip() for col in df.columns]

winter_months = ['december', 'january', 'february']
spring_months = ['april', 'may', 'june']

fake_costs = []

base_inflation = 1.03
winter_bonus = 1.10
spring_consumption_bonus = 1.07
random_noise = 0.07

min_year = df['Year'].min()

for _, row in df.iterrows():
    year = row['Year']
    month = row['Month'].strip().lower()
    salmon_price = row['Salmon/kg prices']
    temperature = row['Temperature UK']
    min_wage = row['Minimum wages UK']

    inflation_factor = base_inflation ** (year - min_year)
    season_factor = 1.0
    if month in winter_months:
        season_factor *= winter_bonus
    if month in spring_months:
        season_factor *= spring_consumption_bonus

    wage_diff = max(salmon_price - min_wage * 10, 1)

    fake_cost = (
        salmon_price * np.random.uniform(90, 110) *
        inflation_factor *
        season_factor +
        wage_diff * np.random.uniform(12, 18) *
        np.random.uniform(1 - random_noise, 1 + random_noise)
    )
    if temperature < 6:
        fake_cost *= 1.03
    if temperature > 22:
        fake_cost *= 0.98

    fake_costs.append(round(fake_cost, 2))

df['Cost'] = fake_costs
df.to_excel('data\\fake_dataset.xlsx', index=False)
print("fake_dataset.xlsx generated.")
