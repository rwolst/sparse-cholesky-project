#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

using namespace std;
using namespace Eigen;

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
}
