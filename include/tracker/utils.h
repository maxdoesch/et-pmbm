#include <Eigen/Dense>

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
        double const _c = 0;
};

struct log_digamma_two_dof
{
    log_digamma_two_dof(double const& w, double const& c) : _w{w}, _c{c} {}

    std::pair<double, double> operator()(double const& x)
    {
        double fx = 2 * _w * std::log(x - 3) - _w * (boost::math::digamma((x - 3) / 2.) + boost::math::digamma((x - 4) / 2.)) + _c;
        double dfx = 2 * _w * 1. / x - _w * (boost::math::trigamma((x - 3) / 2.) + boost::math::trigamma((x - 4) / 2.));

        return std::make_pair(fx, dfx);
    }

    private:
        double const _w;
        double const _c;
};

Eigen::MatrixXd matrixSqrt(Eigen::MatrixXd const& matrix);

double mlgamma(int dim, double num);

void merge_gamma(double& alpha_m, double& beta_m, double const weight[], double const alpha[], double const beta[], int const& components);

void merge_gaussian(Eigen::Vector4d& m_m, Eigen::Matrix4d& P_m, double const weight[], Eigen::Vector4d const m[], Eigen::Matrix4d const P[], int const& components);

void merge_inverse_wishart(double& v_m, Eigen::Matrix2d& V_m, double const weight[], double const v[], Eigen::Matrix2d const V[], int const& components);

double sum_log_weights(double const l_weights[], int const& components);

double sum_log_weights(std::vector<double> const& l_weights);