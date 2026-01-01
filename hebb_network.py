import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("=== PART 1: SIMPLE HEBB EXAMPLE WITH (1,0,1,0) ===")

X = np.array([[1,0,1,0],
              [1,0,1,0]])

y = np.array([1,1])

weights = np.array([1,1,1,1], dtype=float)
bias = 1.0
theta = 2

logs_demo = []

for i in range(len(X)):
    x = X[i]
    target = y[i]

    print("\n---------------------------")
    print("Iteration:", i + 1)
    print("Input X:", x)
    print("Old Weights:", weights)
    print("Old Bias:", bias)
    print("Theta:", theta)

    net = np.dot(weights, x) + bias - theta
    print("Weighted Sum (w·x):", np.dot(weights, x))
    print("Net Input (w·x + b - θ):", net)

    output = 1 if net >= 0 else 0
    print("Predicted Output:", output)
    print("Target:", target)

    delta_w = target * x
    print("Delta W (target * x):", delta_w)

    weights = weights + delta_w
    bias = bias + target

    print("Updated Weights:", weights)
    print("Updated Bias:", bias)

    logs_demo.append([
        x[0], x[1], x[2], x[3],
        delta_w[0], delta_w[1], delta_w[2], delta_w[3],
        weights[0], weights[1], weights[2], weights[3],
        bias
    ])

df_demo = pd.DataFrame(logs_demo, columns=[
    "X1","X2","X3","X4",
    "ΔW1","ΔW2","ΔW3","ΔW4",
    "W1new","W2new","W3new","W4new","Bnew"
])

print("\n=== DEMO HEBB TABLE ===")
print(df_demo)

print("\n=== PART 2: HEBB NETWORK WITH 1000 SAMPLES ===")

class HebbNetwork:
    def __init__(self, num_inputs, learning_rate=1.0):
        self.weights = np.zeros(num_inputs)
        self.bias = 0.0
        self.learning_rate = learning_rate
        print("\nNetwork Initialized")
        print("Weights:", self.weights)
        print("Bias:", self.bias)
        print("Learning Rate:", self.learning_rate)

    def predict(self, x):
        activation = np.dot(x, self.weights) + self.bias
        print("Predict Called")
        print("Input:", x)
        print("Weights:", self.weights)
        print("Bias:", self.bias)
        print("Activation:", activation)
        output = 1 if activation >= 0 else -1
        print("Prediction:", output)
        return output

    def train(self, X, y, epochs=5):
        logs = []
        for epoch in range(epochs):
            print("\n======================================")
            print("Epoch:", epoch + 1)
            print("======================================")
            for i in range(len(X)):
                x = X[i]
                target = y[i]

                print("\nSample:", i + 1)
                print("Input X:", x)
                print("Target:", target)

                old_weights = self.weights.copy()
                old_bias = self.bias

                print("Old Weights:", old_weights)
                print("Old Bias:", old_bias)

                delta_w = self.learning_rate * target * x

                print("Delta W (lr * target * x):", delta_w)

                self.weights = self.weights + delta_w
                self.bias = self.bias + self.learning_rate * target

                print("Updated Weights:", self.weights)
                print("Updated Bias:", self.bias)

                logs.append([
                    x[0], x[1], x[2], x[3],
                    old_weights[0], old_weights[1], old_weights[2], old_weights[3], old_bias,
                    delta_w[0], delta_w[1], delta_w[2], delta_w[3],
                    self.weights[0], self.weights[1], self.weights[2], self.weights[3],
                    self.bias
                ])
        return logs


X, y = make_blobs(n_samples=1000, n_features=4, centers=2, cluster_std=2, random_state=23)

print("\nDataset Generated")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("\nData Scaled")
print("Mean of X:", X.mean())
print("Std of X:", X.std())

y = np.where(y == 0, -1, 1)

print("Labels Converted to {-1, 1}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, shuffle=True)

print("\nTrain-Test Split Done")
print("Train Size:", len(X_train))
print("Test Size:", len(X_test))

hebb = HebbNetwork(num_inputs=4, learning_rate=0.01)

logs = hebb.train(X_train, y_train, epochs=2)

columns = [
    "X1","X2","X3","X4",
    "W1","W2","W3","W4","B",
    "ΔW1","ΔW2","ΔW3","ΔW4",
    "W1new","W2new","W3new","W4new","Bnew"
]

df = pd.DataFrame(logs, columns=columns)

print("\n=== TRAINING LOG (TOP 10 ROWS) ===")
print(df.head(10))

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )

    Z = []
    print("\nGenerating Decision Boundary Predictions")
    for i in range(xx.shape[0]):
        row = []
        for j in range(xx.shape[1]):
            point = np.array([xx[i, j], yy[i, j], 0, 0])
            pred = model.predict(point)
            row.append(pred)
        Z.append(row)

    Z = np.array(Z)

    print("Plotting Decision Boundary")
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Hebbian Network Decision Boundary")
    plt.show()

plot_decision_boundary(hebb, X_test, y_test)
