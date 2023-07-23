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

void GenericValidationModel::draw(cv::Mat& image) const
{
    Eigen::Vector3d kinematicState;
    kinematicState << _k_model->getState().block<2,1>(0,0), _k_model->getState()[4];

    _e_model->draw(image, _color, kinematicState);
}

void GenericValidationModel::print() const
{
    std::cout << "-------" << std::endl;
    std::cout << _k_model->getState() << std::endl;

    std::cout << "-------" << std::endl;
    std::cout << "p_rate: " << _r_model->getRate() << std::endl;
}

Ellipse::Ellipse(double a, double b)  : _a{a}, _b{b} 
{
    
}

void Ellipse::draw(cv::Mat& image, cv::Scalar const& color, Eigen::Vector3d const& state) const
{
    cv::ellipse(image, cv::Point(state[0] * p2co + img_size_x / 2, - state[1] * p2co + img_size_y / 2), cv::Size(_a  * p2co, _b  * p2co), - state[2] * 180 / M_PI, 0, 360, color, stroke_size);
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