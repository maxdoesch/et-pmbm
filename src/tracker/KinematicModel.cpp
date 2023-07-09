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

ConstantVelocity::ConstantVelocity(ConstantVelocity const* k_model)
{
    m = k_model->m;
    P = k_model->P;
    Q = k_model->Q;
    H = k_model->H;
    M = k_model->M;
    G = k_model->G;
}

KinematicModel* ConstantVelocity::copy() const
{
    ConstantVelocity* k_model = new ConstantVelocity(this);

    return k_model;
}

void ConstantVelocity::g(double ts)
{
    G << Eigen::Matrix2d::Identity(), ts * Eigen::Matrix2d::Identity(), Eigen::Matrix2d::Zero(), Eigen::Matrix2d::Identity();

    Q = _sigma * ts * Eigen::Matrix4d::Identity();

    m = G * m;

    P = G * P * G.transpose() + Q;
}

