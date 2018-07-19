# Sparse Cholesky Project
A project that attempts to do the following:

- Wrap the Eigen sparse (and conjugate gradient) Cholesky solver for Python.
- Test speed against the scikits-spase wrapper for CHOLMOD on multiple
  benchmarks.
- Create a Tensorflow op for sparse Cholesky solver.
- Create the Tensorflow forward and backward differentation equations.
