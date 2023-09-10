#pragma once

#include <eigen3/Eigen/Dense>

namespace tracker
{
    class KinematicModel
    {
        public:
            virtual void g(double ts) = 0;

            Eigen::VectorXd m;
            Eigen::MatrixXd P;
            Eigen::MatrixXd Q;
            Eigen::MatrixXd H;
            Eigen::MatrixXd G;
            Eigen::MatrixXd M;
    };

    class ConstantVelocity : public KinematicModel
    {
        public:
            ConstantVelocity();
            ConstantVelocity(ConstantVelocity const& k_model);
            explicit ConstantVelocity(double const weights[], ConstantVelocity const k_models[], int components);
            explicit ConstantVelocity(Eigen::Vector4d const& state, Eigen::Matrix4d const& covariance);
            void operator=(ConstantVelocity const& k_model);

            void g(double ts) override;

        private:
            void _merge(ConstantVelocity& k_model, double const weights[], ConstantVelocity const k_models[], int components);

            double const _sigma = 0.5;
    };
}

