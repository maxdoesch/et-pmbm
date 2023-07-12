#include "tracker/Bernoulli.h"
#include "tracker/utils.h"

using namespace tracker;

Bernoulli::Bernoulli(ExtentModel* e_model) : _e_model{e_model}
{

}

Bernoulli::Bernoulli(Bernoulli const* bernoulli) : _r_model(bernoulli->_r_model)
{
    _p_existence = bernoulli->_p_existence;
    _e_model = bernoulli->_e_model->copy();
}

Bernoulli::~Bernoulli()
{
    delete _e_model;
}

void Bernoulli::predict(double ts)
{
    _p_existence = _p_survival * _p_existence;

    _e_model->predict(ts);
    _r_model.predict();
}

double Bernoulli::misdetection_likelihood()
{  
    double alpha = _r_model.getAlpha();
    double beta = _r_model.getBeta();

    double qd = 1. - _p_detection + _p_detection * pow(beta / (beta + 1), alpha);

    double likelihood = std::log(1 - _p_existence + _p_existence * qd);


    return likelihood;
}

double Bernoulli::detection_likelihood(Cluster const& detection, Bernoulli*& bernoulli)
{
    bernoulli = new Bernoulli(this);

    double update_likelihood = bernoulli->_e_model->update(detection);
    update_likelihood += bernoulli->_r_model.update(detection);
    double bernoulli_likelihood = std::log(_p_existence) + std::log(_p_detection) + update_likelihood;

    bernoulli->_p_existence = 1;

    return bernoulli_likelihood;
}

void Bernoulli::update_misdetection(Bernoulli*& bernoulli)
{
    bernoulli = new Bernoulli(this);

    double alpha = _r_model.getAlpha();
    double beta = _r_model.getBeta();
    double qd = 1. - _p_detection + _p_detection * pow(beta / (beta + 1), alpha);

    bernoulli->_p_existence = bernoulli->_p_existence * qd / (1 - bernoulli->_p_existence + bernoulli->_p_existence * qd);

    double alpha_c[] = {alpha, alpha};
    double beta_c[] = {beta, beta + 1};
    double weight_c[] = {1 / qd * (1 - _p_detection), 1 / qd * _p_detection * pow(beta / (beta + 1), alpha)};

    double alpha_m, beta_m;
    merge_gamma(alpha_m, beta_m, weight_c, alpha_c, beta_c, 2);

    bernoulli->_r_model = RateModel(alpha, beta_m);
}

validation::ValidationModel* Bernoulli::getValidationModel()
{
    return new validation::GenericValidationModel(_e_model->getKinematicValidationModel(), _e_model->getExtentValidationModel(), _r_model.getRateValidationModel());
}