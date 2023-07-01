#include "validation/ValidationModel.h"

using namespace validation;

GenericValidationModel::GenericValidationModel(KinematicModel* k_model, ExtentModel* e_model) 
    : _k_model{k_model}, _e_model{e_model}
{
}

GenericValidationModel::~GenericValidationModel()
{
    delete _k_model;
    delete _e_model;
}

void GenericValidationModel::draw(cv::Mat& image, Parameters const& parameters) const
{
    Eigen::Vector3d kinematicState;
    kinematicState << _k_model->getState().block<2,1>(0,0), _k_model->getState()[4];

    _e_model->draw(image, parameters, kinematicState);
    
    std::cout << "-------" << std::endl;
    std::cout << _k_model->getState() << std::endl;
}


Ellipse::Ellipse(double a, double b, double p_rate) : _a{a}, _b{b}, _p_rate{p_rate}
{

}

void Ellipse::draw(cv::Mat& image, Parameters const& parameters, Eigen::Vector3d const& state) const
{
    cv::ellipse(image, cv::Point(state[0] * parameters._p2co + parameters._img_size_x / 2, - state[1] * parameters._p2co + parameters._img_size_y / 2), cv::Size(_a  * parameters._p2co, _b  * parameters._p2co), - state[2] * 180 / M_PI, 0, 360, cv::Scalar(0, 0, 0));

    std::cout << "-------" << std::endl;
    std::cout << "p_rate: " << _p_rate << std::endl;
}

ConstantVelocity::ConstantVelocity(Eigen::Matrix<double, 5, 1> const& state) : _state{state}
{

}

Eigen::VectorXd ConstantVelocity::getState() const
{
    return _state;
}