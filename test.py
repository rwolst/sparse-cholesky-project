from contexttimer import Timer
from py_eigen.py_eigen_cpp import (matrix_mult_d, matrix_mult_d_blas,
                                   cholesky_dense_d)
import numpy as np
from time import sleep
import sys

np.random.seed(42)

def init(N, M, K):
    A = np.random.rand(N, M)
    B = np.random.rand(M, K)
    C = np.zeros([N, K])
    return A, B, C

def time(N, M, K, problem, method):
    if problem == 'matrix_mult':
        A, B, C = init(N, M, K)
        with Timer() as t:
            if method == 'blas':
                matrix_mult_d_blas(A, B, C)
            elif method == 'eigen':
                matrix_mult_d(A, B, C)
            elif method == 'numpy':
                C = A @ B
        print("%s time: %s" % (method, t.elapsed))
    elif problem == 'cholesky_dense':
        A, _, C = init(N, N, N)
        L = np.tril(A)
        D = np.diag(np.random.rand(N))
        A = L @ D @ L.T + np.eye(N)  # Avoid non-pos def errors.
        A = (A + A.T)/2
        with Timer() as t:
            if method == 'eigen':
                cholesky_dense_d(A, C)
            elif method == 'numpy':
                C = np.linalg.cholesky(A)
        print("%s time: %s" % (method, t.elapsed))

    return C

if __name__ == "__main__":
    C = time(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
             sys.argv[5])

    #print(C)

#C1 = time(1000, 10000, 2000, 'eigen')
#C2 = time(1000, 10000, 2000, 'blas')
#C3 = time(1000, 10000, 2000, 'numpy')
#
#print_values = False
#if print_values:
#    print("Eigen Result")
#    print(C1)
#    print("BLAS Result")
#    print(C2)
#
#    print("A @ B")
#    print(A @ B)
#    print("B @ A")
#    print(B @ A)
