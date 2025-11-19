import numpy as np

def activation(y_in):
    return np.where(y_in > 0, 1, -1)

x = np.array([-1, 1, 1, 1])
print("Input vector (x):", x)

W = np.outer(x, x)
np.fill_diagonal(W, 1)
print("\nWeight matrix W = x^T * x :\n", W)

print("\nTesting with same input vector:")
test = np.array([-1, 1, 1, 1])
yin = np.dot(test, W)
print("Net input (yinj) =", yin)
y = activation(yin)
print("Output (yj) =", y)

print("\nTesting with one missing entry:")
test_cases = [
    np.array([0, 1, 1, 1]),
    np.array([-1, 1, 0, 1])
]
for test in test_cases:
    print("\nTest input:", test)
    yin = np.dot(test, W)
    print("Net input (yinj) =", yin)
    y = activation(yin)
    print("Output (yj) =", y)

print("\nTesting with one mistake entry:")
test = np.array([-1, -1, 1, 1])
print("Test input:", test)
yin = np.dot(test, W)
print("Net input (yinj) =", yin)
y = activation(yin)
print("Output (yj) =", y)

print("\nTesting with two mistaken entries:")
test_cases = [
    np.array([-1, -1, -1, 1]),
    np.array([1, 1, 1, 1])
]
for test in test_cases:
    print("\nTest input:", test)
    yin = np.dot(test, W)
    print("Net input (yinj) =", yin)
    y = activation(yin)
    print("Output (yj) =", y)

print("\nTesting with two missing entries:")
test_cases = [
    np.array([0, 0, 1, 1]),
    np.array([-1, 0, 0, 1])
]
for test in test_cases:
    print("\nTest input:", test)
    yin = np.dot(test, W)
    print("Net input (yinj) =", yin)
    y = activation(yin)
    print("Output (yj) =", y)



import numpy as np

def sign_bipolar(v):
    return np.where(v > 0, 1, np.where(v < 0, -1, 0))

def print_matrix_as_display(v, rows=5, cols=3, name="pattern"):
    mat = v.reshape((rows, cols))
    print(f"{name} (shape {rows}x{cols}):")
    for r in mat:
        print(" ", " ".join(f"{int(x):2d}" for x in r))
    print()

xE = np.array([
     1,  1,  1,
     1, -1, -1,
     1,  1,  1,
     1, -1, -1,
     1,  1,  1
], dtype=int)

xF = np.array([
     1,  1,  1,
     1,  1,  1,
     1, -1, -1,
     1, -1, -1,
     1, -1, -1
], dtype=int)

yE = np.array([-1, 1], dtype=int)
yF = np.array([1, 1], dtype=int)

print("\n Input display patterns (5x3) \n")
print_matrix_as_display(xE, 5, 3, "Input E")
print_matrix_as_display(xF, 5, 3, "Input F")

print("Targets:")
print(" yE =", yE)
print(" yF =", yF)

print("\nBuilding BAM weight matrices\n")
W1 = np.outer(xE, yE)
W2 = np.outer(xF, yF)
W = W1 + W2

np.set_printoptions(linewidth=120, formatter={'int': '{:2d}'.format})
print("W1 = outer(xE, yE):\n", W1)
print("\nW2 = outer(xF, yF):\n", W2)
print("\nW = W1 + W2:\n", W)

def bam_recall(W, x_init=None, y_init=None, max_iters=10, verbose=True):
    Nx, Ny = W.shape
    if x_init is None and y_init is None:
        raise ValueError("Provide x_init or y_init")

    x = None
    y = None
    if x_init is not None:
        x = sign_bipolar(x_init.copy())
    if y_init is not None:
        y = sign_bipolar(y_init.copy())

    for it in range(max_iters):
        changed = False

        if x is not None:
            y_in = x.dot(W)
            y_new = sign_bipolar(y_in)
            if y is None or not np.array_equal(y_new, y):
                changed = True
            y = y_new

        if y is not None:
            x_in = W.dot(y)
            x_new = sign_bipolar(x_in)
            if x is None or not np.array_equal(x_new, x):
                changed = True
            x = x_new

        if not changed:
            break

    return x, y, None

print("\n Test recall: Input E → output y \n")
x_test = xE.copy()
x_rec, y_rec, _ = bam_recall(W, x_init=x_test)
print("Final reconstructed y:", y_rec)
print_matrix_as_display(x_rec, 5, 3, "Final reconstructed x")

print("\n Test recall: Input F → output y \n")
x_test = xF.copy()
x_rec, y_rec, _ = bam_recall(W, x_init=x_test)
print("Final reconstructed y:", y_rec)
print_matrix_as_display(x_rec, 5, 3, "Final reconstructed x")

print("\n Test recall: yE → reconstruct x \n")
x_out, y_out, _ = bam_recall(W, y_init=yE)
print_matrix_as_display(x_out, 5, 3, "x reconstructed from yE")

print("\n Test recall: yF → reconstruct x \n")
x_out, y_out, _ = bam_recall(W, y_init=yF)
print_matrix_as_display(x_out, 5, 3, "x reconstructed from yF")

print("\nTest with noisy input (flip bits of E)\n")
noisyE = xE.copy()
noisy_idxs = [0, 7, 14]
noisyE[noisy_idxs] *= -1
print("Noisy E (flattened):", noisyE.tolist())
print_matrix_as_display(noisyE, 5, 3, "Noisy E")

x_rec, y_rec, _ = bam_recall(W, x_init=noisyE)
print("Reconstructed y:", y_rec)
print_matrix_as_display(x_rec, 5, 3, "Reconstructed x from noisy E")
