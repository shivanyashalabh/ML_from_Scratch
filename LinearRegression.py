# The dataset can be changed as required
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = kagglehub.dataset_download("yasserh/student-marks-dataset")
file_path = os.path.join(path, 'Student_Marks.csv') 
df = pd.read_csv(file_path)
print(df.head())

X = df['time_study'].values  # independent variable
y = df['Marks'].values  # dependent variable

# Normalize input for better training (optional)


class LinearRegressionUsingNumpy:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0
        self.b = 0

    def predict(self, X):
        return self.w * X + self.b

    def fit(self, X, y):
        n = len(X)
        for i in range(self.epochs):
            y_pred = self.predict(X)
            dw = (2/n) * np.sum(X * (y_pred - y))
            db = (2/n) * np.sum(y_pred - y)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            if i % 100 == 0:
                loss = np.mean((y_pred - y) ** 2)
                print(f"Loss at {i}th iteration is {loss:.4f}")

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        print(f"Final MSE is {mse:.4f}")
        return mse

model = LinearRegressionUsingNumpy(learning_rate=0.00001, epochs=10000)
model.fit(X, y)


y_pred = model.predict(X)


plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, color='red', label="Predicted")
plt.title("Predicting Marks from Study Hours")
plt.xlabel("Study Hours (normalized)")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)
plt.show()

model.evaluate(X, y)
print(f"Learned weight (w): {model.w:.2f}")
print(f"Learned bias (b): {model.b:.2f}")
