#include "tracker/ExtentModel.h"
#include "tracker/utils.h"

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/factorials.hpp>

using namespace tracker;

GGIW::GGIW(KinematicModel* k_model) : _k_model{k_model}
{
    _V = Eigen::Matrix2d::Identity();
    _v = 2 * (_dof + 1) + 1; //v > 2d
}

GGIW::GGIW(GGIW const* e_model)
{
    _alpha = e_model->_alpha;
    _beta = e_model->_beta;
    _v = e_model->_v;
    _V = e_model->_V;

    _k_model = e_model->_k_model->copy();
}

GGIW::~GGIW()
{
    delete _k_model;
}

void GGIW::predict(double ts)
{
    _alpha = _alpha / _eta;
    _beta = _beta / _eta;

    _k_model->g(ts);

    _v = 2 * _dof + 2 + std::exp(-ts/_tau) * (_v - 2 * _dof - 2);
    _V = std::exp(- ts / _tau) * _k_model->M * _V * _k_model->M.transpose();
}

double GGIW::update(Cluster const& detection)
{
    Eigen::MatrixXd Z = detection.covariance();
    Eigen::MatrixXd z = detection.mean();
    double n = detection.size();

    Eigen::MatrixXd X_hat = _V / (_v - 2 * _dof - 2);
    Eigen::VectorXd epsilon = z - _k_model->H * _k_model->m;
    Eigen::MatrixXd S = _k_model->H * _k_model->P * _k_model->H.transpose() + X_hat / n;
    Eigen::MatrixXd S_inv = S.inverse();
    Eigen::MatrixXd K = _k_model->P * _k_model->H.transpose() * S_inv;
    
    Eigen::MatrixXd X_hat_sqrt = matrixSqrt(X_hat);
    Eigen::MatrixXd S_inv_sqrt = matrixSqrt(S_inv);

    Eigen::MatrixXd N = X_hat_sqrt * S_inv_sqrt * epsilon * epsilon.transpose() * S_inv_sqrt.transpose() * X_hat_sqrt.transpose();
    
    double alpha = _alpha + n;
    double beta = _beta + 1;

    _k_model->m = _k_model->m + K * epsilon;
    _k_model->P = _k_model->P - K * _k_model->H * _k_model->P;
    double v = _v + n;
    Eigen::MatrixXd V = _V + N + 4 * Z;

    double logLikelihood = - _dof / 2. * (n * std::log(M_PI) + std::log(n));
    logLikelihood += (_v - _dof - 1) / 2. * std::log(_V.determinant()) + mlgamma(_dof, (v - _dof - 1) / 2.) + 0.5 * std::log(X_hat.determinant()) + boost::math::lgamma(alpha) + _alpha * std::log(_beta);
    logLikelihood -= (v - _dof - 1) / 2. * std::log(V.determinant()) + mlgamma(_dof, (_v - _dof - 1) / 2.) + 0.5 * std::log(S.determinant()) + boost::math::lgamma(_alpha) + alpha * std::log(beta);

    _v = v;
    _V = V;
    _alpha = alpha;
    _beta = beta;

    return logLikelihood;
}

double GGIW::getAlpha()
{
    return _alpha;
}

double GGIW::getBeta()
{
    return _beta;
}

void GGIW::setAlpha(double const& alpha)
{
    _alpha = alpha;
}

void GGIW::setBeta(double const& beta)
{
    _beta = beta;
}

validation::ValidationModel* GGIW::getValidationModel()
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigenSolver(_V / (_v - 2 * _dof - 2));
    Eigen::Matrix2d eigenVectors = eigenSolver.eigenvectors();
    Eigen::Vector2d eigenValues = eigenSolver.eigenvalues();

    // Sorting eigenvalues in descending order
    if (eigenValues(0) > eigenValues(1))
    {
        std::swap(eigenValues(0), eigenValues(1));
        eigenVectors.col(0).swap(eigenVectors.col(1));
    }

    // Semi-major and semi-minor axes
    double a = std::sqrt(eigenValues(0));
    double b = std::sqrt(eigenValues(1));

    // Rotation angle (in radians)
    double angle = std::atan2(eigenVectors(1, 0), eigenVectors(0, 0));

    Eigen::Matrix<double, 5, 1> state;
    state << _k_model->m, angle;

    validation::KinematicModel* k_model = new validation::ConstantVelocity(state);
    validation::ExtentModel* e_model = new validation::Ellipse(a, b, _alpha / _beta, CV_RGB(255, 0, 0));
    validation::ValidationModel* v_model = new validation::GenericValidationModel(k_model, e_model);

    return v_model;
}

ExtentModel* GGIW::copy() const
{
    GGIW* e_model = new GGIW(this);

    return e_model;
}