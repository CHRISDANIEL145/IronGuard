import pandas as pd
import numpy as np

# Configuration
ROWS = 100
BASE_TEMP = 1450
BASE_ENERGY = 450

# Generate Data
np.random.seed(42)

# Temperature with some sine wave drift and noise
t = np.linspace(0, 4*np.pi, ROWS)
temps = BASE_TEMP + np.sin(t) * 20 + np.random.normal(0, 5, ROWS)

# Energy correlated with temperature (more heat needed for higher temps usually, but efficiency varies)
# Simplified: Energy = Base + (Temp deviation)^2 * coeff + noise
energy = BASE_ENERGY + ((temps - BASE_TEMP)/10)**2 * 5 + np.random.normal(0, 10, ROWS)

# Add Anomalies (Spikes)
anomaly_indices = [15, 42, 88]
for idx in anomaly_indices:
    temps[idx] += 100  # huge temp spike
    energy[idx] -= 50  # odd energy drop? or spike? let's say drop to indicate sensor failure or efficiency loss

# Create DataFrame
df = pd.DataFrame({
    'temperature': np.round(temps, 1),
    'energy': np.round(energy, 1)
})

# Save
output_path = 'd:\\ironguard\\test_data\\testing_data.csv'
df.to_csv(output_path, index=False)

print(f"✅ Generated {ROWS} rows of synthetic data at: {output_path}")
print(df.head())
