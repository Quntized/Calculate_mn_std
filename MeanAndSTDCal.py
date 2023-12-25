import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Assuming 'prostate.csv' file is present in the specified path
df = pd.read_csv("C:\\Users\\Sajid\\Downloads\\prostate.csv")

# Instantiate StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data, excluding the 'Target' column
scaler.fit(df.drop('Target', axis=1))

# Plot the scaling parameters (mean and standard deviation) for each feature
plt.plot(scaler.mean_, label='Mean')
plt.plot(scaler.scale_, label='Standard Deviation')

# Set labels and legend
plt.xlabel('Feature Index')
plt.ylabel('Scaling Parameter Value')
plt.legend()
plt.show()