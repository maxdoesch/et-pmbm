#pragma once 

#include <vector>
#include <random>
#include <limits>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <eigen3/Eigen/Dense>

#include "validation/ValidationModel.h"

namespace simulator
{
    class Target 
    {
        public:
            virtual ~Target() {}
            virtual void step(double time, pcl::PointCloud<pcl::PointXYZ>::Ptr const& measurements) = 0;
            virtual validation::ValidationModel* getValidationModel() const = 0;
            virtual bool endOfExistence() const = 0;
            virtual bool startOfExistence() const = 0;
    };

    class KinematicModel
    {
        public: 
            virtual void step(double ts, Eigen::Matrix4d& transform) = 0;
            virtual validation::KinematicModel* getKinematicValidationModel() const = 0;

    };   

    class ExtentModel
    {
        public:
            virtual void step(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements) = 0;
            virtual validation::ExtentModel* getExtentValidationModel() const = 0;
            virtual validation::RateModel* getRateValidationModel() const = 0;
    };   

    class GenericTarget : public Target
    {
        public:
            explicit GenericTarget(KinematicModel* k_model, ExtentModel* e_model, double s_o_e, double e_o_e);
            ~GenericTarget();
            void step(double time, pcl::PointCloud<pcl::PointXYZ>::Ptr const& measurements) override;
            validation::ValidationModel* getValidationModel() const override;
            bool endOfExistence() const override;
            bool startOfExistence() const override;

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
            explicit ConstantVelocity(Eigen::Matrix<double, 5, 1> const& initial_state);
            void step(double ts, Eigen::Matrix4d& transform) override;
            validation::KinematicModel* getKinematicValidationModel() const override;

        private:
            Eigen::Matrix<double, 5, 1> _state;
    };

    class Parabola : public KinematicModel
    {
        public:
            explicit Parabola(Eigen::Vector2d const& initial_state, double offset, double time);
            void step(double ts, Eigen::Matrix4d& transform) override;
            validation::KinematicModel* getKinematicValidationModel() const override;

        private:
            Eigen::Matrix<double, 5, 1> _state;

            double const _p = 6;
            int const _sign;
            double const _delta_x;
            double const _delta_alpha;
            double const _a;
            double _offset;
    };

    class Ellipse : public ExtentModel
    {
        public:
            explicit Ellipse(double a, double b, double p_rate);
            void step(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements) override;
            validation::ExtentModel* getExtentValidationModel() const override;
            validation::RateModel* getRateValidationModel() const override;

        private:
            double const _a;
            double const _b;

            double const _p_rate;

            std::random_device _rd;
            std::mt19937 _gen;
    };

    class UniformEllipse : public ExtentModel
    {
        public:
            explicit UniformEllipse(double a, double b, double p_rate);
            void step(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements) override;
            validation::ExtentModel* getExtentValidationModel() const override;
            validation::RateModel* getRateValidationModel() const override;

        private:
            double const _a;
            double const _b;

            double const _p_rate;

            std::random_device _rd;
            std::mt19937 _gen;
    };
}
