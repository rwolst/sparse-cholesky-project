#include <iostream>
#include <ctime>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

using namespace std;
using namespace Eigen;

int matrix_mult_d_c(double *A, double *B, double *C,
        int m, int n, int k) {
    // Convert A, B and C to Eigen arrays.
    Map<Matrix <double, Dynamic, Dynamic, RowMajor> > A_mat(A, m, n);
    Map<Matrix <double, Dynamic, Dynamic, RowMajor> > B_mat(B, n, k);
    Map<Matrix <double, Dynamic, Dynamic, RowMajor> > C_mat(C, m, k);

    // Convert Eigen arrays to matrices.
    // Do the matrix multiply.
    C_mat = (A_mat * B_mat);

    return 0;
}

int matrix_cholesky_d_c(double *A, double *L_T, int m, int n) {
    // Find dense Cholesky factorisation of matrix A and store in triangular
    // matrix L i.e A = L L.T.
    // According to eigen docs, we should use column major for storing lower
    // traingular or row major for storing upper triangular, to avoid 20%
    // slowdown.
    // Note as A is symmetric, it doesn't matter if we use ColMajor when it
    // was stored in RowMajor in Python. Similarly for L, it just affects if
    // we get the lower or upper traingular factor.
    Map<Matrix <double, Dynamic, Dynamic, ColMajor> > A_mat(A, m, n);
    Map<Matrix <double, Dynamic, Dynamic, ColMajor> > L_mat(L_T, m, n);
    LLT<Matrix <double, Dynamic, Dynamic> > llt;
    clock_t start = clock();
    llt.compute(A_mat);
    cout << "LLT time: " << (clock() - start)/(double) CLOCKS_PER_SEC << "s" << endl;
    L_mat = llt.matrixL();

    return 0;
}

int main()
{
    // Create a random lower traingualr matrix.
    MatrixXd A = MatrixXd::Random(5, 5);
    MatrixXd L = A.triangularView<Eigen::Lower>();
    MatrixXd C = L * L.transpose();
    MatrixXd P = C.llt().matrixL();
    SparseMatrix<double> S = C.sparseView();
    SimplicialLLT<SparseMatrix<double> > SLLT;
    SLLT.compute(S);
    if(SLLT.info() != Success) {
      cout << "Decomposition failed";
      return -1;
    }
    cout << "The matrix A:" << endl << A << endl;
    cout << "Lower triangular L:" << endl << L << endl;
    cout << "L * L.T:" << endl << C << endl;
    cout << "Lower factor of L * L.T:" << endl << P << endl;
    cout << "Sparse view of L * L.T:" << endl << S << endl;
    cout << "Sparse lower factor of L * L.T:" << endl << SLLT.matrixL() << endl;


    double A2[] = {1, 2, 3, 4};
    double B2[] = {1, 2, 3, 4};
    double C2[4];
    matrix_mult_d_c(A2, B2, C2, 2, 2, 2);

}
