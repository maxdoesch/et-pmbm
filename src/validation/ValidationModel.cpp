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
    kinematicState << _k_model->state().block<2,1>(0,0), _k_model->state()[4];

    _k_model->draw(image, _color);
    _e_model->draw(image, _color, kinematicState);
}

void GenericValidationModel::draw_position(cv::Mat& image) const
{
    _k_model->draw(image, _color);
}

void GenericValidationModel::print() const
{
    std::cout << "-------" << std::endl;
    std::cout << _k_model->state() << std::endl;

    std::cout << "-------" << std::endl;
    std::cout << "p_rate: " << _r_model->getRate() << std::endl;
}

Eigen::VectorXd GenericValidationModel::state() const
{
    return _k_model->state().block<4, 1>(0,0);
}

Eigen::MatrixXd GenericValidationModel::extent() const
{
    double alpha = _k_model->state()[4];
    Eigen::Matrix2d rot;
    rot << std::cos(alpha), -std::sin(alpha), std::sin(alpha), std::cos(alpha);

    Eigen::Matrix2d X = _e_model->extent();

    return rot * X * rot.transpose();
}

Eigen::VectorXd GenericValidationModel::getExent() const
{
    Eigen::Vector3d extent;
    extent.block<2,1>(0,0) = _e_model->getExtent();
    extent[2] = _k_model->state()[4];

    return extent;
} 

Ellipse::Ellipse(double a, double b)  : _a{a}, _b{b} 
{
    
}

void Ellipse::draw(cv::Mat& image, cv::Scalar const& color, Eigen::Vector3d const& state) const
{
    cv::ellipse(image, cv::Point(state[0] * p2co + img_size_x / 2, - state[1] * p2co + img_size_y / 2), cv::Size(_a  * p2co, _b  * p2co), - state[2] * 180 / M_PI, 0, 360, color, stroke_size);
}

Eigen::MatrixXd Ellipse::extent() const
{
    Eigen::Matrix2d X;
    X << _a*_a, 0, 0, _b*_b;

    return X;
}

Eigen::VectorXd Ellipse::getExtent() const
{
    Eigen::Vector2d extent = {_a, _b};
    
    return extent;
}

ConstantVelocity::ConstantVelocity(Eigen::Matrix<double, 5, 1> const& state) : _state{state}
{

}

void ConstantVelocity::draw(cv::Mat& image, cv::Scalar const& color) const
{
    cv::circle(image, cv::Point(_state[0] * p2co + img_size_x / 2, - _state[1] * p2co + img_size_y / 2), dot_size, color, cv::FILLED);
}

Eigen::VectorXd ConstantVelocity::state() const
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