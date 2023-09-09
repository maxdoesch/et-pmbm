#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "validation/constants.h"


namespace validation
{
    class ValidationModel
    {
        public:
            virtual ~ValidationModel() {}
            virtual void draw(cv::Mat& image) const = 0;
            virtual void draw_position(cv::Mat& image) const = 0;
            virtual void print() const = 0;
            virtual Eigen::VectorXd state() const = 0;
            virtual Eigen::MatrixXd extent() const = 0;
            virtual Eigen::VectorXd getExent() const = 0;
    };

    class ExtentModel
    {
        public:
            virtual void draw(cv::Mat& image, cv::Scalar const& color, Eigen::Vector3d const& state) const = 0;
            virtual Eigen::MatrixXd extent() const = 0;
            virtual Eigen::VectorXd getExtent() const = 0;
    };

    class KinematicModel
    {
        public:
            virtual void draw(cv::Mat& image, cv::Scalar const& color) const = 0;
            virtual Eigen::VectorXd state() const = 0;
    };

    class RateModel
    {
        public:
            explicit RateModel(double p_rate);
            double getRate() const;

        private:
            double _p_rate = 0;
    };

    class GenericValidationModel : public ValidationModel
    {
        public:
            explicit GenericValidationModel(KinematicModel* k_model, ExtentModel* e_model, RateModel* r_model, cv::Scalar const& color);
            GenericValidationModel(GenericValidationModel const& generic_validation_model) = delete;
            ~GenericValidationModel();
            GenericValidationModel& operator=(GenericValidationModel const& generic_validation_model) = delete;

            void draw(cv::Mat& image) const override;
            void draw_position(cv::Mat& image) const override;
            void print() const override;
            Eigen::VectorXd state() const override;
            Eigen::MatrixXd extent() const override;
            Eigen::VectorXd getExent() const override;
            
        private:
            KinematicModel* _k_model;
            ExtentModel* _e_model;
            RateModel* _r_model;

            cv::Scalar _color; 
            
    };

    class Ellipse : public ExtentModel
    {
        public:
            explicit Ellipse(double a, double b);
            Ellipse(Ellipse const& ellipse) = delete;
            Ellipse& operator=(Ellipse const& ellipse) = delete;

            void draw(cv::Mat& image, cv::Scalar const& color, Eigen::Vector3d const& state) const override;
            Eigen::MatrixXd extent() const override;
            Eigen::VectorXd getExtent() const override;

        private:
            double _a = 0;
            double _b = 0;     
    };


    class ConstantVelocity : public  KinematicModel
    {
        public:
            explicit ConstantVelocity(Eigen::Matrix<double, 5, 1> const& state);
            ConstantVelocity(ConstantVelocity const& constant_velocity) = delete;
            ConstantVelocity& operator=(ConstantVelocity const& constant_velocity) = delete;

            void draw(cv::Mat& image, cv::Scalar const& color) const;
            Eigen::VectorXd state() const override;

        private:
            Eigen::Matrix<double, 5, 1> _state; //x, y, dx, dy, alpha
    };
}

