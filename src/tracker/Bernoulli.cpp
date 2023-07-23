#include "tracker/Bernoulli.h"
#include "tracker/utils.h"

#include "tracker/constants.h"

using namespace tracker;

Bernoulli::Bernoulli(ExtentModel* e_model) : _e_model{e_model}
{

}

Bernoulli::Bernoulli(Bernoulli const& bernoulli) :
    _e_model{bernoulli._e_model->copy()}, _r_model(bernoulli._r_model), _p_existence{bernoulli._p_existence}
{
}

Bernoulli::Bernoulli(double p_existence, ExtentModel* e_model, RateModel const& r_model) : _e_model{e_model}, _r_model{r_model}, _p_existence{p_existence}
{

}

Bernoulli::~Bernoulli()
{
    delete _e_model;
}

void Bernoulli::predict(double ts)
{
    _p_existence = p_survival * _p_existence;

    _e_model->predict(ts);
    _r_model.predict();
}

double Bernoulli::missed_detection_likelihood() const
{  
    double alpha = _r_model.getAlpha();
    double beta = _r_model.getBeta();

    double qd = 1. - p_detection + p_detection * pow(beta / (beta + 1), alpha);

    double likelihood = std::log(1 - _p_existence + _p_existence * qd);

    return likelihood;
}

double Bernoulli::detection_likelihood(Cluster const& detection)
{
    double update_likelihood = _e_model->update(detection);
    update_likelihood += _r_model.update(detection);
    double bernoulli_likelihood = std::log(_p_existence) + std::log(p_detection) + update_likelihood;

    _p_existence = 1;

    return bernoulli_likelihood;
}

void Bernoulli::update_missed_detection()
{
    double alpha = _r_model.getAlpha();
    double beta = _r_model.getBeta();
    double qd = 1. - p_detection + p_detection * pow(beta / (beta + 1), alpha);

    _p_existence = _p_existence * qd / (1 - _p_existence + _p_existence * qd);

    double alpha_c[] = {alpha, alpha};
    double beta_c[] = {beta, beta + 1};
    double weight_c[] = {1 / qd * (1 - p_detection), 1 / qd * p_detection * pow(beta / (beta + 1), alpha)};

    double alpha_m, beta_m;
    merge_gamma(alpha_m, beta_m, weight_c, alpha_c, beta_c, 2);

    _r_model = RateModel(alpha_m, beta_m);
}

double Bernoulli::get_pExistence() const
{
    return _p_existence; 
}

validation::ValidationModel* Bernoulli::getValidationModel() const
{
    return new validation::GenericValidationModel(_e_model->getKinematicValidationModel(), _e_model->getExtentValidationModel(), _r_model.getRateValidationModel(), CV_RGB(255, 0, 0));
}

void Bernoulli::operator=(Bernoulli const& bernoulli)
{
    delete _e_model;
    _e_model = bernoulli._e_model->copy();
    _r_model = bernoulli._r_model;
    _p_existence = bernoulli._p_existence;
}

MultiBernoulli::MultiBernoulli(std::vector<Bernoulli> const& bernoullis, double weight) 
    : _bernoullis{bernoullis}, _weight{weight}
{
    
}

MultiBernoulli::MultiBernoulli(MultiBernoulli const& multi_bernoulli) 
    : _bernoullis{multi_bernoulli._bernoullis}, _weight{multi_bernoulli._weight}
{

}

void MultiBernoulli::predict(double ts)
{
    for(auto& bernoulli : _bernoullis)
    {
        bernoulli.predict(ts);
    }
}

void MultiBernoulli::prune(double threshold)
{
    std::vector<tracker::Bernoulli>::iterator bernoulli_iterator;
    for(bernoulli_iterator = _bernoullis.begin(); bernoulli_iterator < _bernoullis.end(); bernoulli_iterator++)
    {
        if((*bernoulli_iterator).get_pExistence() < threshold)
        {
            _bernoullis.erase(bernoulli_iterator);
        }
    }
}

std::vector<Bernoulli> const& MultiBernoulli::getBernoullis() const
{
    return _bernoullis;
}

double MultiBernoulli::getWeight() const
{
    return _weight;
}

void MultiBernoulli::getValidationModels(std::vector<validation::ValidationModel*>& models) const
{
    for(auto const& bernoulli : _bernoullis)
    {
        models.push_back(bernoulli.getValidationModel());
    }
}

void MultiBernoulli::operator=(MultiBernoulli const& multi_bernoulli)
{
    _bernoullis = multi_bernoulli._bernoullis;
    _weight = multi_bernoulli._weight;
}

void MultiBernoulliMixture::add(MultiBernoulli const& multi_bernoulli)
{
    _multi_bernoulli.push_back(multi_bernoulli);
}

MultiBernoulli& MultiBernoulliMixture::operator[](int idx)
{
    return _multi_bernoulli[idx];
}
