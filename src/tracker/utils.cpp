#include "tracker/utils.h"

#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

struct log_digamma
{
    log_digamma(double const& c) : _c{c} {}

    std::pair<double, double> operator()(double const& x)
    {
        double fx = std::log(x) - boost::math::digamma(x) + _c;
        double dfx = 1. / x -  boost::math::trigamma(x);

        return std::make_pair(fx, dfx);
    }

    private:
        double _c = 0;
};


Eigen::MatrixXd matrixSqrt(Eigen::MatrixXd const& matrix)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(matrix);
    Eigen::MatrixXd eigenvalues = eigensolver.eigenvalues().asDiagonal();
    Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
    
    return eigenvectors * eigenvalues.cwiseSqrt() * eigenvectors.transpose();
}

double mlgamma(int dim, double num)
{
    double value = dim * (dim - 1) / 4. * std::log(M_PI);

    for(int i = 1; i <= dim; i++)
    {
        value += boost::math::lgamma(num + (1 - i) / 2.);
    }

    return value;
}

void merge_gamma(double& alpha_m, double& beta_m, double const weight[], double const alpha[], double const beta[], int const& components)
{
    double t_weight = 0;
    double comp_1 = 0;
    double comp_2 = 0;

    double alpha_max = 0;

    for(int i = 0; i < components; i++)
    {
        t_weight += weight[i];

        comp_1 += weight[i] * (boost::math::digamma(alpha[i]) - std::log(beta[i]));
        comp_2 += weight[i] * alpha[i] / beta[i];

        if(alpha[i] > alpha_max)
            alpha_max = alpha[i];
    }

    double bias = 1 / t_weight * comp_1 - std::log(1 / t_weight * comp_2);

    alpha_m = boost::math::tools::newton_raphson_iterate(log_digamma(bias), alpha[0], 0.0, alpha_max * 1.5, 31);

    beta_m = alpha_m / (1 / t_weight * comp_2);
}