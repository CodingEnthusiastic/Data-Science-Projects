A = [0.1, 0.2, 0.4, 0.6, 1.0]
B = [1.0, 0.5, 0.7, 0.3, 0.0]

def fuzzy_union(set1, set2):
    return [round(max(a, b), 2) for a, b in zip(set1, set2)]

def fuzzy_intersection(set1, set2):
    return [round(min(a, b), 2) for a, b in zip(set1, set2)]

def fuzzy_complement(set1):
    return [round(1 - a, 2) for a in set1]

A_union_B = fuzzy_union(A, B)
A_inter_B = fuzzy_intersection(A, B)
A_comp = fuzzy_complement(A)
B_comp = fuzzy_complement(B)
A_union_A = fuzzy_union(A, A_comp)
A_inter_A = fuzzy_intersection(A, A_comp)
B_union_B = fuzzy_union(B, B_comp)
B_inter_B = fuzzy_intersection(B, B_comp)
A_inter_B_comp = fuzzy_intersection(A, B_comp)
A_union_B_comp = fuzzy_union(A, B_comp)
B_union_A = fuzzy_union(B, A_comp)
B_inter_A = fuzzy_intersection(B, A_comp)
A_union_B_comp = fuzzy_complement(fuzzy_union(A, B))
A_inter_B_comp = fuzzy_intersection(A_comp, B_comp)

print("A =", A)
print("B =", B)
print("\n(a) A ∪ B =", A_union_B)
print("(b) A ∩ B =", A_inter_B)
print("(c) A' =", A_comp)
print("(d) B' =", B_comp)
print("(e) A ∪ A' =", A_union_A)
print("(f) A ∩ A' =", A_inter_A)
print("(g) B ∪ B' =", B_union_B)
print("(h) B ∩ B' =", B_inter_B)
print("(i) A ∩ B' =", A_inter_B_comp)
print("(j) A ∪ B' =", A_union_B_comp)
print("(k) B ∩ A' =", B_inter_A)
print("(l) B ∪ A' =", B_union_A)
print("(m) (A ∪ B)' =", A_union_B_comp)
print("(n) (A' ∩ B') =", A_inter_B_comp)

























A = [0.2, 0.3, 0.4, 0.5]
B = [0.1, 0.2, 0.2, 1.0]

def algebraic_sum(A, B):
    return [round((a + b) - (a * b), 2) for a, b in zip(A, B)]

def algebraic_product(A, B):
    return [round(a * b, 2) for a, b in zip(A, B)]

def bounded_sum(A, B):
    return [round(min(1, a + b), 2) for a, b in zip(A, B)]

def bounded_difference(A, B):
    return [round(max(0, a - b), 2) for a, b in zip(A, B)]

alg_sum = algebraic_sum(A, B)
alg_prod = algebraic_product(A, B)
bound_sum = bounded_sum(A, B)
bound_diff = bounded_difference(A, B)

print("A =", A)
print("B =", B)
print("\n(a) Algebraic Sum =", alg_sum)
print"(b) Algebraic Product =", alg_prod)
print("(c) Bounded Sum =", bound_sum)
print("(d) Bounded Difference =", bound_diff)





















A = [0.2, 0.3, 0.4, 0.5]
B = [0.1, 0.2, 0.2, 1.0]

def algebraic_sum(A, B):
    return [round((a + b) - (a * b), 2) for a, b in zip(A, B)]

def algebraic_product(A, B):
    return [round(a * b, 2) for a, b in zip(A, B)]

def bounded_sum(A, B):
    return [round(min(1, a + b), 2) for a, b in zip(A, B)]

def bounded_difference(A, B):
    return [round(max(0, a - b), 2) for a, b in zip(A, B)]

alg_sum = algebraic_sum(A, B)
alg_prod = algebraic_product(A, B)
bound_sum = bounded_sum(A, B)
bound_diff = bounded_difference(A, B)

print("A =", A)
print("B =", B)
print("\n(a) Algebraic Sum =", alg_sum)
print"(b) Algebraic Product =", alg_prod)
print("(c) Bounded Sum =", bound_sum)
print("(d) Bounded Difference =", bound_diff)



import numpy as np
import skfuzzy as fuzz

A = np.array([0.2, 0.3, 0.4, 0.5])
B = np.array([0.1, 0.2, 0.2, 1.0])

algebraic_sum = (A + B) - (A * B)
algebraic_product = A * B
bounded_sum = np.minimum(1, A + B)
bounded_difference = np.maximum(0, A - B)

algebraic_sum = np.round(algebraic_sum, 2)
algebraic_product = np.round(algebraic_product, 2)
bounded_sum = np.round(bounded_sum, 2)
bounded_difference = np.round(bounded_difference, 2)

print("A =", A)
print("B =", B)
print("\n(a) Algebraic Sum =", algebraic_sum)
print"(b) Algebraic Product =", algebraic_product)
print("(c) Bounded Sum =", bounded_sum)
print("(d) Bounded Difference =", bounded_difference)





A = [0.3, 0.7, 1.0]
B = [0.4, 0.9]

def cartesian_product(A, B):
    R = []
    for a in A:
        row = []
        for b in B:
            row.append(round(min(a, b), 2))
        R.append(row)
    return R

def max_min_composition(R, S):
    T = []
    for i in range(len(R)):
        row = []
        for j in range(len(S[0])):
            vals = [min(R[i][k], S[k][j]) for k in range(len(S))]
            row.append(round(max(vals), 2))
        T.append(row)
    return T

def max_product_composition(R, S):
    T = []
    for i in range(len(R)):
        row = []
        for j in range(len(S[0])):
            vals = [R[i][k] * S[k][j] for k in range(len(S))]
            row.append(round(max(vals), 2))
        T.append(row)
    return T

S = [[1, 0.5, 0.3],
     [0.8, 0.4, 0.7]]

R = cartesian_product(A, B)
max_min_T = max_min_composition(R, S)
max_prod_T = max_product_composition(R, S)

print("A =", A)
print("B =", B)
print("\nFuzzy Cartesian Product R:")
for r in R: print(r)

print("\nMax–Min Composition:")
for r in max_min_T: print(r)

print("\nMax–Product Composition:")
for r in max_prod_T: print(r)





import numpy as np
import skfuzzy as fuzz

A = np.array([0.3, 0.7, 1.0])
B = np.array([0.4, 0.9])

R = np.minimum(A[:, None], B[None, :])

S = np.array([
    [1.0, 0.5, 0.3],
    [0.8, 0.4, 0.7]
])

T_maxmin = fuzz.relation_min(R, S)
T_maxprod = fuzz.relation_product(R, S)

np.set_printoptions(precision=2, suppress=True)

print("A =", A)
print("B =", B)
print("\nFuzzy Cartesian Product R:\n", R)
#print("\nMax–Min Composition:\n", T_maxmin)
#print("\nMax–Product Composition:\n", T_maxprod)





import numpy as np

R = np.array([
    [0.6, 0.3],
    [0.8, 0.4]
])

S = np.array([
    [0.8, 0.4, 0.7],
    [0.6, 0.3, 0.9]
])

T_maxmin = np.zeros((R.shape[0], S.shape[1]))
for i in range(R.shape[0]):
    for j in range(S.shape[1]):
        T_maxmin[i, j] = np.max(np.minimum(R[i, :], S[:, j]))

T_maxprod = np.zeros((R.shape[0], S.shape[1]))
for i in range(R.shape[0]):
    for j in range(S.shape[1]):
        T_maxprod[i, j] = np.max(R[i, :] * S[:, j])

np.set_printoptions(precision=2, suppress=True)
print("Max–Min Composition:\n", T_maxmin)
print("\nMax–Product Composition:\n", T_maxprod)





import numpy as np
import skfuzzy as fuzz

R = np.array([
    [0.6, 0.3],
    [0.8, 0.4]
])

S = np.array([
    [0.8, 0.4, 0.7],
    [0.6, 0.3, 0.9]
])

T_maxmin = fuzz.relation_min(R, S)
T_maxprod = fuzz.relation_product(R, S)

np.set_printoptions(precision=2, suppress=True)

print("Fuzzy Relation R:\n", R)
print("\nFuzzy Relation S:\n", S)
print("\nMax–Min Composition:\n", T_maxmin)
print("\nMax–Product Composition:\n", T_maxprod)
