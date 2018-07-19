"""A module exposing some of the eigen functionality to Python through
Cython."""

cimport cython
import numpy as np
cimport numpy as np


cdef extern from "my_program.h":
    cdef int matrix_mult_d_c(double *A, double *B, double *C, int m, int n, int k);
    #cdef int matrix_mult_f_c(float *A, float *B, float *C, int m, int n, int k);
    cdef int matrix_cholesky_d_c(double *A, double *L, int m, int n);

cdef extern from 'cblas.h':
    ctypedef enum CBLAS_LAYOUT:
        CblasRowMajor
        CblasColMajor
    ctypedef enum CBLAS_TRANSPOSE:
        CblasNoTrans
        CblasTrans
        CblasConjTrans

    void dgemm 'cblas_dgemm'(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                             CBLAS_TRANSPOSE TransB, const int M, const int N,
                             const int K, const double alpha, const double *A,
                             const int lda, const double *B, const int ldb,
                             const double beta, double *C, const int ldc) nogil

def matrix_mult_d(double[:,:] A, double[:,:] B, double[:,:] C):
    """Multiply A @ B and store result in C. Where
        A: A m x n matrix.
        B: A n x k matrix.
        C: A m x k matrix.
        """
    m = A.shape[0]
    # assert k.shape[0] == m
    n = A.shape[1]
    # assert B.shape[0] == n
    k = B.shape[1]
    # assert C.shape[1] == k

    matrix_mult_d_c(&A[0,0], &B[0,0], &C[0,0], m, n, k)

def matrix_mult_d_blas(double[:,:] A, double[:,:] B, double[:,:] C):
    """Multiply A @ B and store result in C. Where
        A: A m x n matrix.
        B: A n x k matrix.
        C: A m x k matrix.
        """
    m = A.shape[0]
    # assert k.shape[0] == m
    n = A.shape[1]
    # assert B.shape[0] == n
    k = B.shape[1]
    # assert C.shape[1] == k

    dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          m, n, k, 1.0,
          &A[0,0], n,
          &B[0,0], k,
          1.0,
          &C[0,0], k)

def cholesky_dense_d(double[:,:] A, double[:,:] L):
    """Get lower traingular Cholesky factor of A in L, where
        A: A m x n matrix.
        L: A m x n matrix.
        """
    m = A.shape[0]
    n = A.shape[1]
    # assert L.shape[0] == m
    # assert L.shape[1] == n

    matrix_cholesky_d_c(&A[0,0], &L[0,0], m, n)
