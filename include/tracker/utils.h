#include <Eigen/Dense>

Eigen::MatrixXd matrixSqrt(Eigen::MatrixXd const& matrix);

double mlgamma(int dim, double num);

void merge_gamma(double& alpha_m, double& beta_m, double const weight[], double const alpha[], double const beta[], int const& components);
