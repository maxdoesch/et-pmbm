#include "tracker/utils.h"
#include "tracker/constants.h"

#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include <limits>

using namespace tracker;


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

    double bias = 1. / t_weight * comp_1 - std::log(1. / t_weight * comp_2);

    alpha_m = boost::math::tools::newton_raphson_iterate(log_digamma(bias), alpha[0], 0.0, alpha_max * 1.5, 31);

    beta_m = alpha_m / (1. / t_weight * comp_2);
}

void merge_gaussian(Eigen::Vector4d& m_m, Eigen::Matrix4d& P_m, double const weight[], Eigen::Vector4d const m[], Eigen::Matrix4d const P[], int const& components)
{
    m_m = Eigen::Vector4d::Zero();
    P_m = Eigen::Matrix4d::Zero();
    double t_weight = 0;

    for(int i = 0; i < components; i++)
    {
        m_m += weight[i] * m[i];
        t_weight += weight[i];
    }

    m_m /= t_weight;

    for(int i = 0; i < components; i++)
    {
        Eigen::Vector4d diff = (m[i] - m_m);
        P_m += weight[i] * (P[i] + diff * diff.transpose());
    }

    P_m /= t_weight;
}

void merge_inverse_wishart(double& v_m, Eigen::Matrix2d& V_m, double const weight[], double const v[], Eigen::Matrix2d const V[], int const& components)
{
    double t_weight = 0;
    
    Eigen::Matrix2d comp_1 = Eigen::Matrix2d::Zero();
    double comp_2 = 0;
    double comp_3 = 0;

    double v_max = 0;

    for(int i = 0; i < components; i++)
    {
        t_weight += weight[i];

        comp_1 += weight[i] * (v[i] - 3) * V[i].inverse();
        comp_2 += weight[i] * (boost::math::digamma((v[i] - 3) / 2.) + boost::math::digamma((v[i] - 4) / 2.));
        comp_3 += weight[i] * std::log(V[i].determinant());

        if(v[i] > v_max)
            v_max = v[i];
    }

    double bias = 2 * t_weight * std::log(t_weight) - t_weight * std::log(comp_1.determinant()) + comp_2 - comp_3;

    v_m = boost::math::tools::newton_raphson_iterate(log_digamma_two_dof(t_weight, bias), v[0], 4., v_max * 1.5, 31);

    V_m = t_weight * (v_m - 3) * comp_1.inverse();
}

double sum_log_weights(double const l_weights[], int const& components)
{
    double min_l_weight = std::numeric_limits<double>::infinity();
    long double weight_sum = 0;

    for(int i = 0; i < components; i++)
    {
        if(l_weights[i] < min_l_weight)
            min_l_weight = l_weights[i];
    }

    for(int i = 0; i < components; i++)
    {
        long double l_weight_diff = l_weights[i] - min_l_weight;
        if(l_weight_diff != 0)
        {
            weight_sum += std::exp(l_weight_diff);
        }
    }

    assert(!isinf(weight_sum));

    double l_weight_sum = min_l_weight + std::log(1 + weight_sum);

    return l_weight_sum;
}

double sum_log_weights(std::vector<double> const& l_weights)
{
    double min_l_weight = std::numeric_limits<double>::infinity();
    long double weight_sum = 0;

    for(auto const& l_weight : l_weights)
    {
        if(l_weight < min_l_weight)
            min_l_weight = l_weight;
    }

    for(auto const& l_weight : l_weights)
    {
        long double l_weight_diff = l_weight - min_l_weight;
        if(l_weight_diff != 0)
        {
            weight_sum += std::exp(l_weight_diff);
        } 
    }

    assert(!isinf(weight_sum));

    double l_weight_sum = min_l_weight + std::log(1 + weight_sum);

    return l_weight_sum;
}