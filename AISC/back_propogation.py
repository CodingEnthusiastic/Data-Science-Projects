import math
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def deriv_from_output(y):
    return y * (1.0 - y)

v11 = 0.6
v21 = -0.1
v01 = 0.3
v12 = -0.3
v22 = 0.4
v02 = 0.5

w1 = 0.4
w2 = 0.1
w0 = -0.2

x1, x2 = 0.0, 1.0
target = 1.0
alpha = 0.25
epochs = 10

fmt = lambda v: f"{v:.6f}"

epoch_list, error_list, grad_list = [], [], []

for epoch in range(1, epochs + 1):
    zin1 = v01 + x1 * v11 + x2 * v21
    z1 = sigmoid(zin1)

    zin2 = v02 + x1 * v12 + x2 * v22
    z2 = sigmoid(zin2)

    yin = w0 + z1 * w1 + z2 * w2
    y = sigmoid(yin)

    error = target - y
    fprime_yin = deriv_from_output(y)
    delta_k = error * fprime_yin

    dw1 = alpha * delta_k * z1
    dw2 = alpha * delta_k * z2
    dw0 = alpha * delta_k * 1.0

    delta_in1 = delta_k * w1
    fprime_zin1 = deriv_from_output(z1)
    delta1 = delta_in1 * fprime_zin1

    delta_in2 = delta_k * w2
    fprime_zin2 = deriv_from_output(z2)
    delta2 = delta_in2 * fprime_zin2

    dv11 = alpha * delta1 * x1
    dv21 = alpha * delta1 * x2
    dv01 = alpha * delta1 * 1.0

    dv12 = alpha * delta2 * x1
    dv22 = alpha * delta2 * x2
    dv02 = alpha * delta2 * 1.0

    print("="*72)
    print(f"Epoch {epoch}")
    print("- Forward pass -")
    print(f" Inputs (x1, x2) = ({fmt(x1)}, {fmt(x2)}), Target = {fmt(target)}")
    print(f" Net input z_in1 = {fmt(zin1)}")
    print(f" z1 = {fmt(z1)}")
    print(f" Net input z_in2 = {fmt(zin2)}")
    print(f" z2 = {fmt(z2)}")
    print(f" Net input y_in = {fmt(yin)}")
    print(f" Output y = {fmt(y)}")
    print()
    print("- Error and deltas -")
    print(f" Error (t - y) = {fmt(error)}")
    print(f" f'(y_in) = {fmt(fprime_yin)}")
    print(f" delta_k (output) = {fmt(delta_k)}")
    print()
    print("- Changes in weights (hidden -> output) -")
    print(f" Δw1 = {fmt(dw1)}, Δw2 = {fmt(dw2)}, Δw0 = {fmt(dw0)}")
    print()
    print("- Backprop to hidden layer -")
    print(f" δ1 = {fmt(delta1)}, δ2 = {fmt(delta2)}")
    print()
    print("- Changes in weights (input -> hidden) -")
    print(f" Δv11 = {fmt(dv11)}, Δv21 = {fmt(dv21)}, Δv01 = {fmt(dv01)}")
    print(f" Δv12 = {fmt(dv12)}, Δv22 = {fmt(dv22)}, Δv02 = {fmt(dv02)}")
    print()

    grad_mag = abs(dw1) + abs(dw2) + abs(dw0) + abs(dv11) + abs(dv21) + abs(dv01) + abs(dv12) + abs(dv22) + abs(dv02)
    epoch_list.append(epoch)
    error_list.append(abs(error))
    grad_list.append(grad_mag)

    w1 += dw1
    w2 += dw2
    w0 += dw0

    v11 += dv11
    v21 += dv21
    v01 += dv01

    v12 += dv12
    v22 += dv22
    v02 += dv02

    print("- Updated weights (after epoch) -")
    print(f" w1 = {fmt(w1)}, w2 = {fmt(w2)}, w0 = {fmt(w0)}")
    print(f" v11 = {fmt(v11)}, v21 = {fmt(v21)}, v01 = {fmt(v01)}")
    print(f" v12 = {fmt(v12)}, v22 = {fmt(v22)}, v02 = {fmt(v02)}")
    print("="*72 + "\n\n")

print("\nFinal Error and Gradient Table:")
print("Epoch | Error | Gradient Magnitude")
print("------------------------------------------")
for e, err, g in zip(epoch_list, error_list, grad_list):
    print(f"{e:5d} | {err:.6f} | {g:.6f}")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epoch_list, error_list, marker='o')
plt.title("Error vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Error (|t-y|)")

plt.subplot(1,2,2)
f = lambda w: 0.5 * w**2
w = np.linspace(-3, 3, 400)
E = f(w)
eta = 0.3
w_curr = -2.5
steps = 8
w_hist = [w_curr]
E_hist = [f(w_curr)]

for _ in range(steps):
    grad = w_curr
    w_curr = w_curr - eta * grad
    w_hist.append(w_curr)
    E_hist.append(f(w_curr))

plt.plot(w, E, label="Error Surface E(w)")
plt.plot(w_hist, E_hist, 'ro--', label="Descent Path")
plt.title("Conceptual Gradient Descent (Error vs Weight)")
plt.xlabel("Weight (w)")
plt.ylabel("Error E(w)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
