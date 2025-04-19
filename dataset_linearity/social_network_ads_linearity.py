import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("./datasets/Social_Network_Ads.csv")

# Select 2 features
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Plot with labeled colors
plt.figure(figsize=(6, 6))

# Class 0: Not Purchased
plt.scatter(X[y == 0]['Age'], X[y == 0]['EstimatedSalary'],
            color='blue', label='Not Purchased', edgecolors='k')

# Class 1: Purchased
plt.scatter(X[y == 1]['Age'], X[y == 1]['EstimatedSalary'],
            color='orange', label='Purchased', edgecolors='k')

# Labels and legend
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Social Network Ads Data")
plt.legend()
plt.grid(True)
plt.show()
