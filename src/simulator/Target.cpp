#include "simulator/Target.h"
#include <pcl/common/transforms.h>

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

bool GenericTarget::step(double time, pcl::PointCloud<pcl::PointXYZ>::Ptr const& measurements)
{
    double ts = time - _time;
    _time = time;

    bool end_reached = false;

    if(_time > _start_of_existence && _time < _end_of_existence)
    {
        Eigen::Matrix4d transformation;

        _k_model->step(ts, transformation);
        _e_model->step(measurements);

        pcl::transformPointCloud(*measurements, *measurements, transformation);
    }
    else if(_time > _start_of_existence)
    {
        end_reached = true;
    }

    return end_reached;
}

validation::ValidationModel* GenericTarget::getValidationModel() const
{
    return new validation::GenericValidationModel(_k_model->getValidationModel(), _e_model->getValidationModel());
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

validation::KinematicModel* ConstantVelocity::getValidationModel() const
{
    return new validation::ConstantVelocity(_state);
}


Ellipse::Ellipse(double a, double b, double p_rate) : _a{a}, _b{b}, _p_rate{p_rate}, _gen{_rd()}
{
    _X = Eigen::Matrix2d::Identity();
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

validation::ExtentModel* Ellipse::getValidationModel() const
{
    return new validation::Ellipse(_a, _b, _p_rate, CV_RGB(0, 0, 255));
}