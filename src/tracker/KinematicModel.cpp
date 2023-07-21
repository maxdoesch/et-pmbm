#include "tracker/KinematicModel.h"
#include <iostream>
using namespace tracker;

ConstantVelocity::ConstantVelocity()
{
    m = Eigen::Vector4d::Zero();
    P = _sigma * Eigen::Matrix4d::Identity();
    Q = Eigen::Matrix4d::Zero();
    H = Eigen::Matrix<double, 2, 4>::Identity();
    M = Eigen::Matrix2d::Identity();
    G = Eigen::Matrix4d::Identity();
}

ConstantVelocity::ConstantVelocity(ConstantVelocity const& k_model)
{
    m = k_model.m;
    P = k_model.P;
    Q = k_model.Q;
    H = k_model.H;
    M = k_model.M;
    G = k_model.G;
}

ConstantVelocity::ConstantVelocity(double const weights[], ConstantVelocity const k_models[], int components) : ConstantVelocity()
{
    _merge(*this, weights, k_models, components);
}

ConstantVelocity::ConstantVelocity(Eigen::Vector4d const& state, Eigen::Matrix4d const& covariance)
{
    m = state;
    P = covariance;
    Q = Eigen::Matrix4d::Zero();
    H = Eigen::Matrix<double, 2, 4>::Identity();
    M = Eigen::Matrix2d::Identity();
    G = Eigen::Matrix4d::Identity();
}

void ConstantVelocity::g(double ts)
{
    G << Eigen::Matrix2d::Identity(), ts * Eigen::Matrix2d::Identity(), Eigen::Matrix2d::Zero(), Eigen::Matrix2d::Identity();

    Q = _sigma * ts * Eigen::Matrix4d::Identity();

    m = G * m;

    P = G * P * G.transpose() + Q;
}

void ConstantVelocity::operator=(ConstantVelocity const& k_model)
{
    m = k_model.m;
    P = k_model.P;
    Q = k_model.Q;
    H = k_model.H;
    M = k_model.M;
    G = k_model.G;
}

void ConstantVelocity::_merge(ConstantVelocity& k_model, double const weights[], ConstantVelocity const k_models[], int components)
{
    k_model.m = Eigen::Vector4d::Zero();
    k_model.P = Eigen::Matrix4d::Zero();
    double t_weight = 0;

    for(int i = 0; i < components; i++)
    {
        k_model.m += weights[i] * k_models[i].m;
        t_weight += weights[i];
    }

    k_model.m /= t_weight;

    for(int i = 0; i < components; i++)
    {
        Eigen::Vector4d diff = (k_models[i].m - k_model.m);
        k_model.P += weights[i] * (k_models[i].P + diff * diff.transpose());
    }

    k_model.P /= t_weight;
}