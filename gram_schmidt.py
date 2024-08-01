"""
Gram Schmidt Orthoganalization
and
QR Decomposition
"""


import random
import numpy as np



def proj(a, q):
    """Project vector a upon vector q"""
    a, q = np.array(a), np.array(q)
    return np.dot(q, a) / np.dot(q, q) * q


def normalize(v):
    m = sum(v*v for v in v) ** (1/2)
    return [v/m for v in v]


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


def is_orthogonal(matrix):
    return np.allclose(np.dot(matrix, matrix.T), np.eye(len(matrix)))


def qr_decomposition(A):
    Q = gram_schmidt(A)
    R = np.dot(Q.T, A)
    return Q,R


def inverse(A):
    """Easier inversion of A, because the upper triangular matrix R is easier to inverse"""
    Q,R = qr_decomposition(A)
    return np.dot(np.linalg.inv(R), Q.T)


if __name__ == '__main__':
    
    # Demo on Gram Schmidt orthoganalization
    A = random_matrix()
    Q = gram_schmidt(A)
    print("Q is orthonormal:", is_orthogonal(Q), "\n")
    print(A, Q.round(2), sep="\n\n")
    
    
    # Demo on QR decomposition
    Q, R = qr_decomposition(A)
    print("\n\n\nmatreces A, Q, R, A-restored:", A, Q.round(2), R.round(2), np.dot(Q,R).round(1), sep="\n\n")
    
