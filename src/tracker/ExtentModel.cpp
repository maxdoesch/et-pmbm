#include "tracker/ExtentModel.h"
#include "tracker/utils.h"

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/factorials.hpp>

using namespace tracker;

template class GIW<ConstantVelocity>;

template <class KinematicTemplate> GIW<KinematicTemplate>::GIW()
{
    _V = Eigen::Matrix2d::Identity();
    _v = 2 * (_dof + 1) + 1; //v > 2d
}

template <class KinematicTemplate> GIW<KinematicTemplate>::GIW(GIW const* e_model) : _k_model(e_model->_k_model)
{
    _v = e_model->_v;
    _V = e_model->_V;
}

template <class KinematicTemplate> GIW<KinematicTemplate>::~GIW()
{

}

template <class KinematicTemplate> void GIW<KinematicTemplate>::predict(double ts)
{
    _k_model.g(ts);

    _v = 2 * _dof + 2 + std::exp(-ts/_tau) * (_v - 2 * _dof - 2);
    _V = std::exp(- ts / _tau) * _k_model.M * _V * _k_model.M.transpose();
}

template <class KinematicTemplate> double GIW<KinematicTemplate>::update(Cluster const& detection)
{
    Eigen::MatrixXd Z = detection.covariance();
    Eigen::MatrixXd z = detection.mean();
    double n = detection.size();

    Eigen::MatrixXd X_hat = _V / (_v - 2 * _dof - 2);
    Eigen::VectorXd epsilon = z - _k_model.H * _k_model.m;
    Eigen::MatrixXd S = _k_model.H * _k_model.P * _k_model.H.transpose() + X_hat / n;
    Eigen::MatrixXd S_inv = S.inverse();
    Eigen::MatrixXd K = _k_model.P * _k_model.H.transpose() * S_inv;
    
    Eigen::MatrixXd X_hat_sqrt = matrixSqrt(X_hat);
    Eigen::MatrixXd S_inv_sqrt = matrixSqrt(S_inv);

    Eigen::MatrixXd N = X_hat_sqrt * S_inv_sqrt * epsilon * epsilon.transpose() * S_inv_sqrt.transpose() * X_hat_sqrt.transpose();

    _k_model.m = _k_model.m + K * epsilon;
    _k_model.P = _k_model.P - K * _k_model.H * _k_model.P;
    double v = _v + n;
    Eigen::MatrixXd V = _V + N + 4 * Z;

    double logLikelihood = - _dof / 2. * (n * std::log(M_PI) + std::log(n));
    logLikelihood += (_v - _dof - 1) / 2. * std::log(_V.determinant()) + mlgamma(_dof, (v - _dof - 1) / 2.) + 0.5 * std::log(X_hat.determinant());
    logLikelihood -= (v - _dof - 1) / 2. * std::log(V.determinant()) + mlgamma(_dof, (_v - _dof - 1) / 2.) + 0.5 * std::log(S.determinant());

    _v = v;
    _V = V;

    return logLikelihood;
}

template <class KinematicTemplate> validation::ExtentModel* GIW<KinematicTemplate>::getExtentValidationModel() const
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

    return new validation::Ellipse(a, b, CV_RGB(255, 0, 0));
}

template <class KinematicTemplate> validation::KinematicModel* GIW<KinematicTemplate>::getKinematicValidationModel() const
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

    // Rotation angle (in radians)
    double angle = std::atan2(eigenVectors(1, 0), eigenVectors(0, 0));

    Eigen::Matrix<double, 5, 1> state;
    state << _k_model.m, angle;

    return new validation::ConstantVelocity(state);
}

template <class KinematicTemplate> ExtentModel* GIW<KinematicTemplate>::copy() const
{
    GIW* e_model = new GIW(this);

    return e_model;
}

RateModel::RateModel() {};

RateModel::RateModel(double alpha, double beta) : _alpha{alpha}, _beta{beta} {}

RateModel::RateModel(RateModel const& r_model) : _alpha{r_model._alpha}, _beta{r_model._beta} {}

void RateModel::predict()
{
    _alpha = _alpha / _eta;
    _beta = _beta / _eta;
}

double RateModel::update(Cluster const& detection)
{
    double n = detection.size();

    double alpha = _alpha + n;
    double beta = _beta + 1;


    double log_likelihood = boost::math::lgamma(alpha) + _alpha * std::log(_beta);
    log_likelihood -= boost::math::lgamma(_alpha) + alpha * std::log(beta);

    _alpha = alpha;
    _beta = beta;

    return log_likelihood;
}

double RateModel::getAlpha()
{
    return _alpha;
}

double RateModel::getBeta()
{
    return _beta;
}

double RateModel::getRate()
{
    return _alpha / _beta;
}

RateModel RateModel::operator=(const tracker::RateModel& r_model) const
{
    return r_model;
}

validation::RateModel* RateModel::getRateValidationModel() const
{
    return new validation::RateModel(_alpha / _beta);
}