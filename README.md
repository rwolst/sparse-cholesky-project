# Sparse Cholesky Project
A project that attempts to do the following:

- Wrap the Eigen sparse (and conjugate gradient) Cholesky solver for Python.
- Test speed against the scikits-spase wrapper for CHOLMOD on multiple
  benchmarks.
- Create a Tensorflow op for sparse Cholesky solver.
- Create the Tensorflow forward and backward differentation equations.

## Installation
### Eigen
To install Eigen, simply download the source from [here](http://eigen.tuxfamily.org/index.php?title=Main_Page)
and copy the header files into a location that is in the PATH e.g.

    cd ~/Downloads
    tar -xvf ~/eigen-eigen-<build-info>
    cd eigen-eigen-<build-info>
    sudo cp -r Eigen /usr/local/include
