#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>


namespace validation
{
    struct Parameters
    {
        int const _point_size = 2;
        int const _img_size_x = 1280;
        int const _img_size_y = 720;
        int const _p2co = 50;
    };

    class ValidationModel
    {
        public:
            virtual void draw(cv::Mat& image, Parameters const& parameters) const = 0;
    };

    class ExtentModel
    {
        public: 
            virtual void draw(cv::Mat& image, Parameters const& parameters, Eigen::Vector3d const& state) const = 0;
    };

    class KinematicModel
    {
        public:
            virtual Eigen::VectorXd getState() const = 0;
    };

    class GenericValidationModel : public ValidationModel
    {
        public:
            GenericValidationModel(KinematicModel* k_model, ExtentModel* e_model);
            ~GenericValidationModel();
            void draw(cv::Mat& image, Parameters const& parameters) const override;

        private:
            KinematicModel* _k_model;
            ExtentModel* _e_model;
            
    };

    class Ellipse : public ExtentModel
    {
        public:
            Ellipse(double a, double b, double p_rate);
            void draw(cv::Mat& image, Parameters const& parameters, Eigen::Vector3d const& state) const override;

        private:
            double _a = 0;
            double _b = 0;
            double _p_rate = 0;            
    };


    class ConstantVelocity : public  KinematicModel
    {
        public:
            ConstantVelocity(Eigen::Matrix<double, 5, 1> const& state);
            Eigen::VectorXd getState() const override;

        private:
            Eigen::Matrix<double, 5, 1> _state; //x, y, dx, dy, alpha
    };
}

