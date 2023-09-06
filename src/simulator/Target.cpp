#include "simulator/Target.h"
#include <pcl/common/transforms.h>
#include <cmath>

using namespace simulator;

GenericTarget::GenericTarget(KinematicModel* k_model, ExtentModel* e_model, double s_o_e, double e_o_e) 
    : _k_model{k_model}, _e_model{e_model}, _time{s_o_e}, _start_of_existence{s_o_e}, _end_of_existence{e_o_e}
{

}

GenericTarget::~GenericTarget()
{
    delete _k_model;
    delete _e_model;
}

void GenericTarget::step(double time, pcl::PointCloud<pcl::PointXYZ>::Ptr const& measurements)
{
    double ts = time - _time;
    _time = time;

    if(_time >= _start_of_existence && _time < _end_of_existence)
    {
        Eigen::Matrix4d transformation;

        _k_model->step(ts, transformation);
        _e_model->step(measurements);

        pcl::transformPointCloud(*measurements, *measurements, transformation);
    }
}

validation::ValidationModel* GenericTarget::getValidationModel() const
{
    return new validation::GenericValidationModel(_k_model->getKinematicValidationModel(), _e_model->getExtentValidationModel(), _e_model->getRateValidationModel(), CV_RGB(0, 0, 255));
}

bool GenericTarget::endOfExistence() const
{
    return _time > _end_of_existence;
}

bool GenericTarget::startOfExistence() const
{
    return _time >= _start_of_existence;
}

ConstantVelocity::ConstantVelocity(Eigen::Matrix<double, 5, 1> const& initial_state) : _state{initial_state}
{

}

void ConstantVelocity::step(double ts, Eigen::Matrix4d& transformation)
{
    Eigen::Matrix4d motionModel;
    motionModel << Eigen::Matrix2d::Identity(), ts * Eigen::Matrix2d::Identity(), Eigen::Matrix2d::Zero(), Eigen::Matrix2d::Identity();
    _state.block<4,1>(0,0) = motionModel * _state.block<4,1>(0,0);

    Eigen::Matrix3d rotation;
    rotation << std::cos(_state[4]), -std::sin(_state[4]), 0, std::sin(_state[4]), std::cos(_state[4]), 0, 0, 0, 1;
    Eigen::Vector3d translation;
    translation << _state.block<2, 1>(0, 0), 0;
    transformation << rotation, translation, Eigen::Matrix<double, 1, 3>::Zero(), 1;
}

validation::KinematicModel* ConstantVelocity::getKinematicValidationModel() const
{
    return new validation::ConstantVelocity(_state);
}

Parabola::Parabola(Eigen::Vector2d const& initial_state,  double offset, double time) : 
    _sign{(initial_state[1] > 0) - (initial_state[1] < 0)}, _delta_x{-2 * initial_state[0] / time}, _delta_alpha{-_sign * M_PI / (2 * time)}, _a{std::pow(((std::abs(initial_state[1]) - offset) / std::pow(initial_state[0], _p)), 1. / _p)}, _offset{_sign * offset}
{
    _state = Eigen::Matrix<double, 5, 1>::Zero();
    _state.block<2,1>(0,0) = initial_state;
    _state[4] = -_sign * M_PI / 4;
}

void Parabola::step(double ts, Eigen::Matrix4d& transformation)
{
    double x =  _state[0] + _delta_x * ts;
    double y = _sign * std::pow(_a * x, _p) + _offset;
    double dx = _delta_x;
    double dy = (y - _state[1]) / ts;
    double alpha = _state[4] + _delta_alpha * ts;

    _state << x, y, dx, dy, alpha;

    Eigen::Matrix3d rotation;
    rotation << std::cos(_state[4]), -std::sin(_state[4]), 0, std::sin(_state[4]), std::cos(_state[4]), 0, 0, 0, 1;
    Eigen::Vector3d translation;
    translation << _state.block<2, 1>(0, 0), 0;
    transformation << rotation, translation, Eigen::Matrix<double, 1, 3>::Zero(), 1;
}

validation::KinematicModel* Parabola::getKinematicValidationModel() const
{
    return new validation::ConstantVelocity(_state);
}


Ellipse::Ellipse(double a, double b, double p_rate) : _a{a}, _b{b}, _p_rate{p_rate}, _gen{_rd()}
{

}

void Ellipse::step(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements)
{
    std::poisson_distribution<> poisson(_p_rate); 
    std::normal_distribution<> normal(0, 1);

    /*
    Eigen::Matrix2d rot;
    rot << std::cos(alpha), -std::sin(alpha), std::sin(alpha), std::cos(alpha);
    Eigen::Matrix2d dim;
    dim << _a *_a, 0, 0, _b *_b;

    _X = rot * dim * rot.transpose();

    Eigen::LLT<Eigen::MatrixXd> cholSolver(_X);
    Eigen::Matrix2d normal_transform;
    normal_transform = cholSolver.matrixL();

    int samples = poisson(_gen);
    for(int i = 0; i < samples; i++)
    {
        Eigen::Vector2d sample;
        sample << normal(_gen), normal(_gen);
        sample = normal_transform * sample;

        pcl::PointXYZ point;
        point.x = sample[0];
        point.y = sample[1];

        measurements->push_back(point);
    }*/

    int samples = poisson(_gen);
    for(int i = 0; i < samples; i++)
    {
        pcl::PointXYZ point;
        point.x = 0.5 * _a * normal(_gen);
        point.y = 0.5 * _b * normal(_gen);

        measurements->push_back(point);
    }
}

validation::ExtentModel* Ellipse::getExtentValidationModel() const
{
    return new validation::Ellipse(_a, _b);
}

validation::RateModel* Ellipse::getRateValidationModel() const
{
    return new validation::RateModel(_p_rate);
}

UniformEllipse::UniformEllipse(double a, double b, double p_rate) : _a{a}, _b{b}, _p_rate{p_rate}, _gen{_rd()}
{

}

void UniformEllipse::step(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements)
{
    std::poisson_distribution<> poisson(_p_rate); 
    std::uniform_real_distribution<> theta_uniform(0, 2 * M_PI);
    std::uniform_real_distribution<> r_uniform(0, 1);

    int samples = poisson(_gen);
    for(int i = 0; i < samples; i++)
    {
        pcl::PointXYZ point;
        double theta = theta_uniform(_gen);
        double r = std::sqrt(r_uniform(_gen));

        point.x = _a * r * std::cos(theta);
        point.y = _b * r * std::sin(theta);

        measurements->push_back(point);
    }
}

validation::ExtentModel* UniformEllipse::getExtentValidationModel() const
{
    return new validation::Ellipse(_a, _b);
}

validation::RateModel* UniformEllipse::getRateValidationModel() const
{
    return new validation::RateModel(_p_rate);
}