#include "validation/ValidationModel.h"

using namespace validation;

GenericValidationModel::GenericValidationModel(KinematicModel* k_model, ExtentModel* e_model, RateModel* r_model, cv::Scalar const& color) 
    : _k_model{k_model}, _e_model{e_model}, _r_model{r_model}, _color{color}
{
}

GenericValidationModel::~GenericValidationModel()
{
    delete _k_model;
    delete _e_model;
    delete _r_model;
}

void GenericValidationModel::draw(cv::Mat& image, Parameters const& parameters) const
{
    Eigen::Vector3d kinematicState;
    kinematicState << _k_model->getState().block<2,1>(0,0), _k_model->getState()[4];

    std::cout << "-------" << std::endl;
    std::cout << _k_model->getState() << std::endl;

    std::cout << "-------" << std::endl;
    std::cout << "p_rate: " << _r_model->getRate() << std::endl;

    _e_model->draw(image, parameters, _color, kinematicState);
}

Ellipse::Ellipse(double a, double b)  : _a{a}, _b{b} 
{
    
}

void Ellipse::draw(cv::Mat& image, Parameters const& parameters, cv::Scalar const& color, Eigen::Vector3d const& state) const
{
    cv::ellipse(image, cv::Point(state[0] * parameters._p2co + parameters._img_size_x / 2, - state[1] * parameters._p2co + parameters._img_size_y / 2), cv::Size(_a  * parameters._p2co, _b  * parameters._p2co), - state[2] * 180 / M_PI, 0, 360, color);
}

ConstantVelocity::ConstantVelocity(Eigen::Matrix<double, 5, 1> const& state) : _state{state}
{

}

Eigen::VectorXd ConstantVelocity::getState() const
{
    return _state;
}

RateModel::RateModel(double p_rate) : _p_rate{p_rate}
{

}

double RateModel::getRate() const
{
    return _p_rate;
}