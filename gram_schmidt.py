"""Gram Schmidt Orthoganalization"""


import random
import numpy as np



def proj(a, q):
    """Project vector a upon vector q"""
    a, q = np.array(a), np.array(q)
    return (np.dot(q, a) / np.dot(q,q)) * q


def normalize(v):
    m = sum(v*v for v in v) ** (1/2)
    return [v/m for v in v]


def is_orthonormal(matrix):
    matrix = np.array(matrix)
    return np.allclose(np.dot(matrix, matrix.T), np.eye(len(matrix)))


def random_matrix():
    m = n = random.randint(1, 10)
    A = np.random.randint(0, 10, size=(m,n))
    return A.astype(np.float_)


def gram_schmidt(A):
    """Make matrix A orthornomal via Gram Schmidt"""
    Q = np.zeros_like(A)
    for j in range(A.shape[1]):   # A.shape[1] = n
        Q[:,j] = normalize(A[:,j] - sum(proj(A[:,j], Q[:,k]) for k in range(j)))
    return Q


# Demo
A = random_matrix()
Q = gram_schmidt(A)
print("Q is orthonormal:", is_orthonormal(Q), "\n")
print(A, Q.round(2), sep="\n\n")

