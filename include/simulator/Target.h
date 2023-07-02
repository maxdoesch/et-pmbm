#pragma once 

#include <vector>
#include <random>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

#include "validation/ValidationModel.h"

namespace simulator
{
    class Target 
    {
        public:
            virtual bool step(double time, pcl::PointCloud<pcl::PointXYZ>::Ptr const& measurements) = 0;
            virtual validation::ValidationModel* getValidationModel() const = 0;
    };

    class KinematicModel
    {
        public: 
            virtual void step(double ts, Eigen::Matrix4d& transform) = 0;
            virtual validation::KinematicModel* getValidationModel() const = 0;

    };   

    class ExtentModel
    {
        public:
            virtual void step(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements) = 0;
            virtual validation::ExtentModel* getValidationModel() const = 0;
    };   

    class GenericTarget : public Target
    {
        public:
            GenericTarget(KinematicModel* k_model, ExtentModel* e_model, double s_o_e, double e_o_e);
            ~GenericTarget();
            bool step(double time, pcl::PointCloud<pcl::PointXYZ>::Ptr const& measurements) override;
            validation::ValidationModel* getValidationModel() const override;

        private:
            KinematicModel* _k_model;
            ExtentModel* _e_model;

            double _time = 0;

            double const _start_of_existence = 0;
            double const _end_of_existence = std::numeric_limits<double>::infinity();
    };

    class ConstantVelocity : public KinematicModel
    {
        public:
            ConstantVelocity(Eigen::Matrix<double, 5, 1> const& initial_state);
            void step(double ts, Eigen::Matrix4d& transform) override;
            validation::KinematicModel* getValidationModel() const override;

        private:
            Eigen::Matrix<double, 5, 1> _state;
    };

    class Ellipse : public ExtentModel
    {
        public:
            Ellipse(double a, double b, double p_rate);
            void step(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements) override;
            validation::ExtentModel* getValidationModel() const override;

        private:
            Eigen::Matrix2d _X;

            double const _a;
            double const _b;

            double const _p_rate;

            std::random_device _rd;
            std::mt19937 _gen;
    };
}
