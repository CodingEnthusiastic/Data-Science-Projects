import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_step(x, theta=2):
    return np.where(x >= theta, 1, 0)

def and_gate(x1, x2):
    X = np.array([x1, x2])
    W = np.array([1, 1])
    z = np.dot(X, W)
    print("AND Gate Input:", X, "Weighted Sum:", z)
    return binary_step(z)

def half_adder(x1, x2):
    X = np.array([x1, x2])
    print("\nHalf Adder Inputs:", X)

    W_sum = np.array([1, -1])
    z_sum = np.dot(X, W_sum)
    sigmoid_sum = sigmoid(z_sum)
    sum_out = int(np.round(sigmoid_sum))
    print("Sum Weighted Sum:", z_sum, "Sigmoid:", sigmoid_sum, "Rounded:", sum_out)

    W_carry = np.array([1, 1])
    z_carry = np.dot(X, W_carry)
    sigmoid_carry = sigmoid(z_carry)
    carry_out = int(np.round(sigmoid_carry))
    print("Carry Weighted Sum:", z_carry, "Sigmoid:", sigmoid_carry, "Rounded:", carry_out)

    return sum_out, carry_out, z_sum, sigmoid_sum, z_carry, sigmoid_carry

def full_adder(x1, x2, cin):
    print("\nFull Adder Inputs:", x1, x2, cin)

    s1, c1, z1, sig1, z2, sig2 = half_adder(x1, x2)
    s2, c2, z3, sig3, z4, sig4 = half_adder(s1, cin)

    final_sum = s2
    final_carry = int(np.round(sigmoid(c1 + c2)))

    print("Intermediate Sums:", s1, s2)
    print("Intermediate Carries:", c1, c2)
    print("Final Sum:", final_sum, "Final Carry:", final_carry)

    return final_sum, final_carry, z1, sig1, z2, sig2, z3, sig3, z4, sig4


half_adder_results = []
print("\nHalf Adder (Sigmoid Activation)")
print("X1 X2 | z(Sum) | σ(z) | z(Carry) | σ(z) | Sum Carry")

for x1 in [0, 1]:
    for x2 in [0, 1]:
        s, c, z_sum, sig_sum, z_carry, sig_carry = half_adder(x1, x2)
        half_adder_results.append((x1, x2, s, c))
        print(f"{x1} {x2} | {z_sum:6.2f} | {sig_sum:.4f} | {z_carry:7.2f} | {sig_carry:.4f} | {s:^3} {c}")

full_adder_results = []
print("\nFull Adder (Sigmoid Activation)")
print("X1 X2 Cin | Sum Carry")

for x1 in [0, 1]:
    for x2 in [0, 1]:
        for cin in [0, 1]:
            s, co, z1, sig1, z2, sig2, z3, sig3, z4, sig4 = full_adder(x1, x2, cin)
            full_adder_results.append((x1, x2, cin, s, co))
            print(f"{x1} {x2} {cin} | {s} {co}")

print("\nSummary: Half Adder Truth Table")
print("X1 X2 | Sum Carry")
for x1, x2, s, c in half_adder_results:
    print(f"{x1} {x2} | {s} {c}")

print("\nSummary: Full Adder Truth Table")
print("X1 X2 Cin | Sum Carry")
for x1, x2, cin, s, c in full_adder_results:
    print(f"{x1} {x2} {cin} | {s} {c}")
