#include "tracker/ExtentModel.h"
#include "tracker/utils.h"

#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/factorials.hpp>

using namespace tracker;

template class GIW<ConstantVelocity>;

template <class KinematicTemplate> GIW<KinematicTemplate>::GIW()
{
    _V = Eigen::Matrix2d::Identity();
    _v = 2 * (_dof + 1) + 1; //v > 2d
    _X_hat = _V / (_v - 2 * _dof - 2);
}

template <class KinematicTemplate> GIW<KinematicTemplate>::GIW(GIW const* e_model) : _k_model(e_model->_k_model)
{
    _v = e_model->_v;
    _V = e_model->_V;
    _X_hat = e_model->_X_hat;
}

template <class KinematicTemplate> GIW<KinematicTemplate>::GIW(double const weights[], GIW<KinematicTemplate> const e_models[], int components)
{
    _merge(*this, weights, e_models, components);

    if(_v < 2 * (_dof + 1))
        _selectMostLikely(*this, weights, e_models, components);

    _X_hat = _V / (_v - 2 * _dof - 2);

    KinematicTemplate* k_models = new KinematicTemplate[components];
    for(int i = 0; i < components; i++)
    {
        k_models[i] = e_models[i]._k_model;
    }
    _k_model = KinematicTemplate(weights, k_models, components);

    delete[] k_models;
}

template <class KinematicTemplate> GIW<KinematicTemplate>::GIW(std::vector<double> const& weights, std::vector<GIW> const& e_models)
{
    _merge(weights, e_models);

    if(_v < 2 * (_dof + 1))
        _selectMostLikely(weights, e_models);

    _X_hat = _V / (_v - 2 * _dof - 2);
}

template <class KinematicTemplate> GIW<KinematicTemplate>::GIW(Eigen::Vector4d const& m, Eigen::Matrix4d const& P, Eigen::Matrix2d const& V, double v) : 
    _k_model(m, P)
{
    _V = V;
    _v = (v > 2 * _dof) ? v : 2 * (_dof + 1) + 1; //v > 2d
    _X_hat = _V / (_v - 2 * _dof - 2);
}

template <class KinematicTemplate> GIW<KinematicTemplate>::~GIW()
{

}

template <class KinematicTemplate> void GIW<KinematicTemplate>::operator=(GIW const& e_model)
{
    _v = e_model._v;
    _V = e_model._V;
    _X_hat = e_model._X_hat;
    _k_model = e_model._k_model;
}

template <class KinematicTemplate> void GIW<KinematicTemplate>::predict(double ts)
{
    _k_model.g(ts);

    double v = _v;

    _v = 2 * _dof + 2 + std::exp(-ts/_tau) * (_v - 2 * _dof - 2);
    _V = std::exp(- ts / _tau) * _k_model.M * _V * _k_model.M.transpose();
    _X_hat = _V / (_v - 2 * _dof - 2);
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
    _X_hat = _V / (_v - 2 * _dof - 2);

    return logLikelihood;
}

template <class KinematicTemplate> double GIW<KinematicTemplate>::squared_distance(Cluster const& detection) const
{
    Eigen::Matrix2d R = _X_hat + _k_model.H * _k_model.P * _k_model.H.transpose();
    Eigen::Vector2d diff = detection.mean() - _k_model.H * _k_model.m;

    Eigen::Matrix2d R_2 = (R + 4 * detection.covariance() / detection.size()) / 2.;

    double squared_distance = diff.transpose() * R_2.inverse() * diff;

    return squared_distance;
}

template <class KinematicTemplate> validation::ExtentModel* GIW<KinematicTemplate>::getExtentValidationModel() const
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigenSolver(_X_hat);
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

    return new validation::Ellipse(a, b);
}

template <class KinematicTemplate> validation::KinematicModel* GIW<KinematicTemplate>::getKinematicValidationModel() const
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigenSolver(_X_hat);
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

template <class KinematicTemplate> void GIW<KinematicTemplate>::_merge(GIW<KinematicTemplate>& e_model, double const weights[], GIW<KinematicTemplate> const e_models[], int components)
{
    double t_weight = 0;
    
    Eigen::Matrix2d comp_1 = Eigen::Matrix2d::Zero();
    double comp_2 = 0;
    double comp_3 = 0;

    double v_max = 0;

    for(int i = 0; i < components; i++)
    {
        t_weight += weights[i];

        comp_1 += weights[i] * (e_models[i]._v - 3) * e_models[i]._V.inverse();
        comp_2 += weights[i] * (boost::math::digamma((e_models[i]._v - 3) / 2.) + boost::math::digamma((e_models[i]._v - 4) / 2.));
        comp_3 += weights[i] * std::log(e_models[i]._V.determinant());

        if(e_models[i]._v > v_max)
            v_max = e_models[i]._v;
    }

    double bias = 2 * t_weight * std::log(t_weight) - t_weight * std::log(comp_1.determinant()) + comp_2 - comp_3;

    double v_min = 2 * _dof + 2;
    e_model._v = boost::math::tools::newton_raphson_iterate(log_digamma_two_dof(t_weight, bias), e_models[0]._v, 4., v_max * 1.5, 31);
    e_model._V = t_weight * (e_model._v - 3) * comp_1.inverse();
}

template <class KinematicTemplate> void GIW<KinematicTemplate>::_merge(std::vector<double> const& weights, std::vector<GIW> const& e_models)
{
    double t_weight = 0;
    Eigen::Matrix2d comp_1 = Eigen::Matrix2d::Zero();
    double comp_2 = 0;
    double comp_3 = 0;
    double v_max = 0;

    _k_model = KinematicTemplate();

    auto weights_it = weights.begin();
    auto e_models_it = e_models.begin();
    for(; weights_it < weights.end();)
    {
        t_weight += *weights_it;

        comp_1 += *weights_it * (e_models_it->_v - 3) * e_models_it->_V.inverse();
        comp_2 += *weights_it * (boost::math::digamma((e_models_it->_v - 3) / 2.) + boost::math::digamma((e_models_it->_v - 4) / 2.));
        comp_3 += *weights_it * std::log(e_models_it->_V.determinant());

        if(e_models_it->_v > v_max)
            v_max = e_models_it->_v;

        _k_model.m += *weights_it * e_models_it->_k_model.m;

        ; weights_it++; 
        e_models_it++;
    }

    double bias = 2 * t_weight * std::log(t_weight) - t_weight * std::log(comp_1.determinant()) + comp_2 - comp_3;

    double v_min = 2 * _dof + 2;
    _v = boost::math::tools::newton_raphson_iterate(log_digamma_two_dof(t_weight, bias), e_models[0]._v, 4., v_max * 1.5, 31);
    _V = t_weight * (_v - 3) * comp_1.inverse();

    _k_model.m /= t_weight;

    weights_it = weights.begin();
    e_models_it = e_models.begin();
    for(; weights_it < weights.end();)
    {
        Eigen::VectorXd diff = e_models_it->_k_model.m - _k_model.m;
        _k_model.P += *weights_it * (e_models_it->_k_model.P + diff * diff.transpose());

        weights_it++; 
        e_models_it++;
    }

    _k_model.P /= t_weight;
}

template <class KinematicTemplate> void  GIW<KinematicTemplate>::_selectMostLikely(GIW& e_model, double const weights[], GIW const e_models[], int components)
{
    double max_weight = 0;

    for(int i = 0; i < components; i++)
    {
        if(weights[i] > max_weight)
        {
            max_weight = weights[i];
            e_model._v = e_models[i]._v;
            e_model._V = e_models[i]._V;
        }
    }
}

template <class KinematicTemplate> void GIW<KinematicTemplate>::_selectMostLikely(std::vector<double> const& weights, std::vector<GIW> const& e_models)
{
    double max_weight = 0;

    auto weights_it = weights.begin();
    auto e_models_it = e_models.begin();
    for(; weights_it < weights.end();)
    {
        if(*weights_it > max_weight)
        {
            max_weight = *weights_it;
            _v = e_models_it->_v;
            _V = e_models_it->_V;
        }

        weights_it++;
        e_models_it++;
    }
}

RateModel::RateModel() {};

RateModel::RateModel(double alpha, double beta) : _alpha{alpha}, _beta{beta} {}

RateModel::RateModel(RateModel const& r_model) : _alpha{r_model._alpha}, _beta{r_model._beta} {}

RateModel::RateModel(double const weights[], RateModel const r_models[], int components)
{
    _merge(*this, weights, r_models, components);
}

RateModel::RateModel(std::vector<double> const& weights, std::vector<RateModel> const& r_models)
{
    _merge(weights, r_models);
}

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

double RateModel::getAlpha() const
{
    return _alpha;
}

double RateModel::getBeta() const
{
    return _beta;
}

double RateModel::getRate() const
{
    return _alpha / _beta;
}

void RateModel::operator=(RateModel const& r_model)
{
    _alpha = r_model._alpha;
    _beta = r_model._beta;
}

validation::RateModel* RateModel::getRateValidationModel() const
{
    return new validation::RateModel(_alpha / _beta);
}

void RateModel::_merge(RateModel& r_model_m, double const weights[], RateModel const r_models[], int const& components) const
{
    double t_weight = 0;
    double comp_1 = 0;
    double comp_2 = 0;

    double alpha_max = 0;

    for(int i = 0; i < components; i++)
    {
        t_weight += weights[i];

        comp_1 += weights[i] * (boost::math::digamma(r_models[i]._alpha) - std::log(r_models[i]._beta));
        comp_2 += weights[i] * r_models[i]._alpha / r_models[i]._beta;

        if(r_models[i]._alpha > alpha_max)
            alpha_max = r_models[i]._alpha;
    }

    double bias = 1 / t_weight * comp_1 - std::log(1 / t_weight * comp_2);

    r_model_m._alpha = boost::math::tools::newton_raphson_iterate(log_digamma(bias), r_models[0]._alpha, 0.0, alpha_max * 1.5, 31);

    r_model_m._beta = r_model_m._alpha / (1 / t_weight * comp_2);
}

void RateModel::_merge(std::vector<double> const& weights, std::vector<RateModel> const& r_models)
{
    double t_weight = 0;
    double comp_1 = 0;
    double comp_2 = 0;

    double alpha_max = 0;

    auto weights_it = weights.begin();
    auto r_models_it = r_models.begin();
    for(; weights_it < weights.end();)
    {
        t_weight += *weights_it;

        comp_1 += *weights_it * (boost::math::digamma(r_models_it->_alpha) - std::log(r_models_it->_beta));
        comp_2 += *weights_it * r_models_it->_alpha / r_models_it->_beta;

        if(r_models_it->_alpha > alpha_max)
            alpha_max = r_models_it->_alpha;

        weights_it++; 
        r_models_it++;
    }

    double bias = 1 / t_weight * comp_1 - std::log(1 / t_weight * comp_2);

    _alpha = boost::math::tools::newton_raphson_iterate(log_digamma(bias), r_models[0]._alpha, 0.0, alpha_max * 1.5, 31);

    _beta = _alpha / (1 / t_weight * comp_2);
}