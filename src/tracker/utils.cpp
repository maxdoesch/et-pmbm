#include "tracker/utils.h"

Eigen::MatrixXd matrixSqrt(Eigen::MatrixXd const& matrix)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(matrix);
    Eigen::MatrixXd eigenvalues = eigensolver.eigenvalues().asDiagonal();
    Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
    
    return eigenvectors * eigenvalues.cwiseSqrt() * eigenvectors.transpose();
}

double mlgamma(int dim, double num)
{
    double value = dim * (dim - 1) / 4 * log(M_PI);

    for(int i = 1; i <= dim; i++)
    {
        value += std::lgamma(num + (1 - i) / 2);
    }

    return value;
}