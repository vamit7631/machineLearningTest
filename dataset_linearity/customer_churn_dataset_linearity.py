import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./datasets/customer_churn_dataset.csv")

print(df.columns)

X = df[['age', 'tenure']]
y = df['churn']  # This is the label (0 or 1)

plt.figure(figsize=(6,6))
plt.scatter(X[y == 0]['age'], X[y == 0]['tenure'], color="blue", label="Not Churn" , edgecolors='k')
plt.scatter(X[y == 1]['age'], X[y == 1]['tenure'], color="red", label="Churn" , edgecolors='k')
plt.xlabel("Age")
plt.ylabel("Tenure")
plt.title("Raw Data Linearity Visualization (No Model)")
plt.legend()
plt.grid(True)
plt.show()