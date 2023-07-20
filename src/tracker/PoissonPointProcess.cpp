#include "tracker/PoissonPointProcess.h"
#include "tracker/utils.h"

#include "constants.h"

using namespace tracker;

PPP::~PPP()
{
    for(PoissonComponent* component : _p_components)
        delete component;
}

void PPP::predict(double ts)
{
    for(auto component : _p_components)
        component->predict(ts);

    _b_model.birth(_p_components);
}

void PPP::update_missed_detection()
{
    for(auto component : _p_components)
    {
        component->update_missed_detection();
    }
}

double PPP::detection_likelihood(Cluster const& detection, Bernoulli*& bernoulli) const
{
    int total_components = _p_components.size();
    GIW<ConstantVelocity>* e_models = new GIW<ConstantVelocity>[total_components];
    RateModel* r_models = new RateModel[total_components];
    double* l_weights = new double[total_components];
    double* weights = new double[total_components];

    int idx = 0;
    for(auto component : _p_components)
    {
        l_weights[idx]  = component->detection_likelihood(detection, e_models[idx], r_models[idx]);
        l_weights[idx]  += std::log(component->getWeight()) + std::log(p_detection);

        idx++;
    }

    double l_weight_sum = sum_log_weights(l_weights, total_components);

    for(int i = 0; i < total_components; i++)
    {
        l_weights[i] -= l_weight_sum;
        weights[i] = std::exp(l_weights[i]);
    }

    GIW<ConstantVelocity>* e_model = new GIW<ConstantVelocity>(weights, e_models, total_components);
    RateModel r_model(weights, r_models, total_components);

    double p_existence = 1;
    
    if(detection.size() == 1)
        p_existence = 1. / (std::exp(std::log(clutter_rate) - l_weight_sum) + 1);

    bernoulli = new Bernoulli(p_existence, e_model, r_model);

    delete[] e_models;
    delete[] r_models;
    delete[] l_weights;
    delete[] weights;

    return l_weight_sum;
}

void PPP::getValidationModels(std::vector<validation::ValidationModel*>& models)
{
    for(PoissonComponent* component : _p_components)
        models.push_back(component->getValidationModel());
}

PoissonComponent::PoissonComponent(double weight, GIW<ConstantVelocity> const& e_model, RateModel const& r_model) : _e_model{e_model},  _weight{weight}, _r_model{r_model}
{

}


PoissonComponent::PoissonComponent(PoissonComponent const& p_component) :  _e_model{p_component._e_model}, _r_model{p_component._r_model}, _weight{p_component._weight}
{

}

PoissonComponent::~PoissonComponent() 
{

}

void PoissonComponent::predict(double ts)
{
    _weight = _weight * p_survival;

    _e_model.predict(ts);
    _r_model.predict();
}

void PoissonComponent::update_missed_detection()
{
    double alpha = _r_model.getAlpha();
    double beta = _r_model.getBeta();

    double weight_1 = (1 - p_detection) * _weight;
    double weight_2 = p_detection * pow(beta / (beta + 1), alpha) * _weight;

    double alphas[] = {alpha, alpha};
    double betas[] = {beta, beta + 1};
    double weights[] = {weight_1, weight_2};

    double alpha_m, beta_m;
    merge_gamma(alpha_m, beta_m, weights, alphas, betas, 2);

    _r_model = RateModel(alpha_m, beta_m);
    _weight = weight_1 + weight_2;
}

double PoissonComponent::detection_likelihood(Cluster const& detection, GIW<ConstantVelocity>& e_model, RateModel& r_model) const
{
    e_model = _e_model;
    r_model = _r_model;

    double update_likelihood = e_model.update(detection);
    update_likelihood += r_model.update(detection);

    return update_likelihood;
}

double PoissonComponent::getWeight() const
{
    return _weight;
}

validation::ValidationModel* PoissonComponent::getValidationModel() const
{
    return new validation::GenericValidationModel(_e_model.getKinematicValidationModel(), _e_model.getExtentValidationModel(), _r_model.getRateValidationModel(), CV_RGB(0, 255, 0));
}

BirthModel::BirthModel()
{
    Eigen::Matrix4d init_state_covariance = Eigen::Matrix4d::Zero();
    init_state_covariance(0, 0) = _field_of_view_x / (_n_components + 1);
    init_state_covariance(1, 1) = _field_of_view_y / (_n_components + 1);

    Eigen::Matrix2d init_extent_matrix = Eigen::Matrix2d::Zero(); 
    init_extent_matrix(0, 0) = _field_of_view_x / (_n_components + 1);
    init_extent_matrix(1, 1) = _field_of_view_y / (_n_components + 1);

    for(int i = 0; i < _n_components; i++)
    {
        double x = _field_of_view_x / (_n_components + 1) * (i + 1) - _field_of_view_x / 2.;

        for(int j = 0; j < _n_components; j++)
        {
            double y = _field_of_view_y / (_n_components + 1) * (j + 1) - _field_of_view_y / 2.;

            Eigen::Vector4d init_state = Eigen::Vector4d::Zero();
            init_state[0] = x;
            init_state[1] = y;

            GIW<ConstantVelocity> e_model(init_state, init_state_covariance, init_extent_matrix);
            RateModel r_model(50, 5);
            _birth_components.push_back(PoissonComponent(1. / (_n_components * _n_components), e_model, r_model));
        }
    }
}

BirthModel::~BirthModel()
{
}

void BirthModel::birth(std::vector<PoissonComponent*>& b_components) const
{
    for(auto& birth_component : _birth_components)
    {
        b_components.push_back(new PoissonComponent(birth_component));
    }
}