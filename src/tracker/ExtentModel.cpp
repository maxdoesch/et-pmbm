#include "tracker/ExtentModel.h"
#include "tracker/utils.h"

using namespace tracker;

GGIW::GGIW(KinematicModel* k_model) : _k_model{k_model}
{
    _V = Eigen::Matrix2d::Identity();
    _v = 2 * (_dof + 1) + 1; //v > 2d
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
    Eigen::MatrixXd X_hat = _V / (_v - 2 * _dof - 2);
    Eigen::VectorXd epsilon = detection.mean() - _k_model->H * _k_model->m;
    Eigen::MatrixXd S = _k_model->H * _k_model->P * _k_model->H.transpose() + X_hat / detection.size();
    Eigen::MatrixXd S_inv = S.inverse();
    Eigen::MatrixXd K = _k_model->P * _k_model->H.transpose() * S_inv;
    
    Eigen::MatrixXd X_hat_sqrt = matrixSqrt(X_hat);
    Eigen::MatrixXd S_inv_sqrt = matrixSqrt(S_inv);

    Eigen::MatrixXd N = X_hat_sqrt * S_inv_sqrt * epsilon * epsilon.transpose() * S_inv_sqrt.transpose() * X_hat_sqrt.transpose();
    
    double alpha = _alpha + detection.size();
    double beta = _beta + 1;

    _k_model->m = _k_model->m + K * epsilon;
    _k_model->P = _k_model->P - K * _k_model->H * _k_model->P;
    double v = _v + detection.size();
    Eigen::MatrixXd V = _V + N + detection.covariance();

    double logLikelihood = - _dof / 2. * (detection.size() * std::log(M_PI) + std::log(detection.size()));
    logLikelihood += (_v - _dof - 1) / 2. * std::log(_V.determinant()) + mlgamma(_dof, (v - _dof - 1) / 2.) + 0.5 * std::log(X_hat.determinant()) + lgamma(alpha) + _alpha * std::log(_beta);
    logLikelihood -= (v - _dof - 1) / 2. * std::log(V.determinant()) + mlgamma(_dof, (_v - _dof - 1) / 2.) + 0.5 * std::log(S.determinant()) + lgamma(_alpha) + alpha * std::log(beta);

    _v = v;
    _V = V;
    _alpha = alpha;
    _beta = beta;

    std::cout << "ll: " << logLikelihood << std::endl;

    return logLikelihood;
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
    validation::ExtentModel* e_model = new validation::Ellipse(a, b, _alpha / _beta);
    validation::ValidationModel* v_model = new validation::GenericValidationModel(k_model, e_model);

    return v_model;
}