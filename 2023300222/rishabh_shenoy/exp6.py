import numpy as np

inputs = np.array([
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1]
])

weights = np.array([
    [0.2, 0.4, 0.6, 0.8],
    [0.9, 0.7, 0.5, 0.3]
])

learning_rate = 0.5
epochs = 10

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

for epoch in range(epochs):

    print(f"\n--- Epoch {epoch + 1} ---")
    if epoch == 0:
        print("Learning rate updated to :", learning_rate)

    for idx, input_vector in enumerate(inputs):
        print(f"\nInput Vector {idx + 1}: {input_vector}")

        D1 = euclidean_distance(input_vector, weights[0]) * euclidean_distance(input_vector, weights[0])
        D2 = euclidean_distance(input_vector, weights[1]) * euclidean_distance(input_vector, weights[1])

        print(f"Euclidean Distances -> D1: {D1:.4f}, D2: {D2:.4f}")

        winner = np.argmin([D1, D2])
        print(f"Winning Neuron: Neuron {winner + 1} with Distance: {min(D1, D2):.4f}")

        weights[winner] = weights[winner] + learning_rate * (input_vector - weights[winner])

        print(f"Updated Weights after training with Input {idx + 1}:")
        print(f"Neuron 1 Weights: {weights[0]}")
        print(f"Neuron 2 Weights: {weights[1]}")

    learning_rate = learning_rate * learning_rate

    if epoch < 9:
        print("Learning rate updated to :", learning_rate)
