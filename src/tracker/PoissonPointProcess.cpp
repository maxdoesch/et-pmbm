#include "tracker/PoissonPointProcess.h"
#include "tracker/Bernoulli.h"
#include "tracker/utils.h"

#include "tracker/constants.h"

using namespace tracker;

PPP::PPP() 
{

}

PPP::PPP(PPP const& ppp) : _p_components{ppp._p_components}, _b_model{ppp._b_model}
{

}

PPP::~PPP()
{

}

PPP& PPP::operator=(PPP const& ppp)
{
    _p_components = ppp._p_components;
    _b_model = ppp._b_model;

    return *this;
}

void PPP::predict(double ts)
{
    for(auto& component : _p_components)
        component.predict(ts);

    _b_model.birth(_p_components);
}

void PPP::update_missed_detection()
{
    for(auto& component : _p_components)
    {
        component.update_missed_detection();
    }
}

void PPP::prune(double threshold)
{
    std::vector<tracker::PoissonComponent>::iterator p_components_iterator;
    for(p_components_iterator = _p_components.begin(); p_components_iterator < _p_components.end();)
    {
        if((*p_components_iterator).getWeight() < threshold)
        {
            p_components_iterator = _p_components.erase(p_components_iterator);
        }
        else
            p_components_iterator++;
    }
}

void PPP::capping(int N)
{
    std::priority_queue<PoissonComponent, std::vector<PoissonComponent>, std::greater<PoissonComponent>> minHeap;

    for(auto const& _p_component : _p_components)
    {
        if(minHeap.size() < N)
            minHeap.push(_p_component);
        else if(_p_component.getWeight() > minHeap.top().getWeight())
        {
            minHeap.pop();
            minHeap.push(_p_component);
        }
    }

    _p_components.clear();
    _p_components.reserve(minHeap.size());

    while(!minHeap.empty())
    {
        _p_components.push_back(minHeap.top());
        minHeap.pop();
    }
}

void PPP::add_component(PoissonComponent const& p_component)
{
    _p_components.push_back(p_component);
}

Bernoulli PPP::detection_likelihood(Cluster const& detection, double& likelihood) const
{
    int n_components = _p_components.size();
    std::vector<GIW<ConstantVelocity>> e_models;
    std::vector<RateModel> r_models;
    std::vector<double> l_weights;
    std::vector<double> n_weights;
    e_models.reserve(n_components);
    r_models.reserve(n_components);
    l_weights.reserve(n_components);
    n_weights.reserve(n_components);

    for(auto const& component : _p_components)
    {
        GIW<ConstantVelocity> e_model;
        RateModel r_model;
        double l_weight = component.detection_likelihood(detection, e_model, r_model);
        l_weight += std::log(component.getWeight()) + std::log(p_detection);

        e_models.push_back(e_model);
        r_models.push_back(r_model);
        l_weights.push_back(l_weight);
    }

    double l_weight_sum = sum_log_weights(l_weights);
    
    for(auto& l_weight : l_weights)
    {
        l_weight -= l_weight_sum;
        double weight = std::exp(l_weight);
        n_weights.push_back(weight);
    }

    //prune unlikely associations
    auto e_models_it = e_models.begin();
    auto r_models_it = r_models.begin();
    auto l_weights_it = l_weights.begin();
    auto n_weights_it = n_weights.begin();

    for(; e_models_it < e_models.end();)
    {
        if((*n_weights_it) < _min_likelihood)
        {
            e_models_it = e_models.erase(e_models_it);
            r_models_it = r_models.erase(r_models_it);
            l_weights_it = l_weights.erase(l_weights_it);
            n_weights_it = n_weights.erase(n_weights_it);
        }
        else
        {
            e_models_it++;
            r_models_it++;
            l_weights_it++;
            n_weights_it++;
        }
    }

    GIW<ConstantVelocity>* e_model = new GIW<ConstantVelocity>(n_weights, e_models);
    RateModel r_model(n_weights, r_models);

    double p_existence = 1;
    
    if(detection.size() < 1)
        p_existence = 1. / (std::exp(std::log(clutter_rate) - l_weight_sum) + 1);

    likelihood = l_weight_sum;

    Bernoulli bernoulli(p_existence, e_model, r_model);
    return bernoulli;
}

void PPP::getValidationModels(std::vector<validation::ValidationModel*>& models)
{
    for(auto const& component : _p_components)
        models.push_back(component.getValidationModel());
}


PoissonComponent::PoissonComponent(double weight, GIW<ConstantVelocity> const& e_model, RateModel const& r_model) : _e_model{e_model},  _weight{weight}, _r_model{r_model}
{

}

PoissonComponent::PoissonComponent(double l_weight, Bernoulli const& bernoulli) : 
    _e_model{*dynamic_cast<GIW<ConstantVelocity>*>(bernoulli._e_model)}, _r_model{bernoulli._r_model}, _weight{std::exp(l_weight) * bernoulli._p_existence}
{

}


PoissonComponent::PoissonComponent(PoissonComponent const& p_component) :  _e_model{p_component._e_model}, _r_model{p_component._r_model}, _weight{p_component._weight}
{

}

PoissonComponent::~PoissonComponent() 
{

}

PoissonComponent& PoissonComponent::operator=(PoissonComponent const& p_component)
{
    _e_model = p_component._e_model;
    _r_model = p_component._r_model;
    _weight = p_component._weight;

    return *this;
}

void PoissonComponent::predict(double ts)
{
    _weight *= p_survival;

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

bool PoissonComponent::operator>(PoissonComponent const& p_component) const
{
    return _weight > p_component._weight;
}

BirthModel::BirthModel()
{
    Eigen::Matrix4d P = Eigen::Matrix4d::Zero();
    P(0, 0) = pow(field_of_view_x / (3 * (_n_components + 1)), 2);
    P(1, 1) = pow(field_of_view_y / (3 * (_n_components + 1)), 2);

    Eigen::Matrix2d V = _V_rad * Eigen::Matrix2d::Identity(); 
    
    for(int i = 0; i < _n_components; i++)
    {
        double x = field_of_view_x / (_n_components + 1) * (i + 1) - field_of_view_x / 2.;

        for(int j = 0; j < _n_components; j++)
        {
            double y = field_of_view_y / (_n_components + 1) * (j + 1) - field_of_view_y / 2.;

            Eigen::Vector4d m = Eigen::Vector4d::Zero();
            m[0] = x;
            m[1] = y;

            GIW<ConstantVelocity> e_model(m, P, V, _v);
            RateModel r_model(_alpha, _beta);
            _birth_components.push_back(PoissonComponent(_weight / (_n_components * _n_components), e_model, r_model));
        }
    }
}

BirthModel::BirthModel(BirthModel const& birth_model) : _birth_components{birth_model._birth_components}
{

}

BirthModel::~BirthModel()
{

}

BirthModel& BirthModel::operator=(BirthModel const& birth_model)
{
    _birth_components = birth_model._birth_components;

    return *this;
}

void BirthModel::birth(std::vector<PoissonComponent>& b_components) const
{
    for(auto& birth_component : _birth_components)
    {
        b_components.push_back(PoissonComponent(birth_component));
    }
}

CenterBirthModel::CenterBirthModel()
{
    Eigen::Vector4d m = Eigen::Vector4d::Zero();
    m[0] = 0;
    m[1] = 0;

    Eigen::Matrix4d P = Eigen::Matrix4d::Zero();
    P(0, 0) = pow(field_of_view_x / 3, 2);
    P(1, 1) = pow(field_of_view_y / 3, 2);
    P(2,2) = 0.01;
    P(3,3) = 0.01;

    Eigen::Matrix2d V = _V_rad * Eigen::Matrix2d::Identity(); 

    GIW<ConstantVelocity> e_model(m, P, V, _v);
    RateModel r_model(_alpha, _beta);
    _birth_components.push_back(PoissonComponent(1, e_model, r_model));
}

CenterBirthModel::CenterBirthModel(CenterBirthModel const& birth_model) : _birth_components{birth_model._birth_components}
{

}

CenterBirthModel::~CenterBirthModel()
{

}

CenterBirthModel& CenterBirthModel::operator=(CenterBirthModel const& birth_model)
{
    _birth_components = birth_model._birth_components;

    return *this;
}

void CenterBirthModel::birth(std::vector<PoissonComponent>& b_components) const
{
    for(auto& birth_component : _birth_components)
    {
        b_components.push_back(PoissonComponent(birth_component));
    }
}