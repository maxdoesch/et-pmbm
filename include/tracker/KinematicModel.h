#pragma once

#include <Eigen/Dense>

namespace tracker
{
    class KinematicModel
    {
        public:
            virtual KinematicModel* copy() const = 0;
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
            ConstantVelocity(ConstantVelocity const* k_model);
            KinematicModel* copy() const;
            void g(double ts) override;

        private:
            double const _sigma = 0.01;
    };
}

