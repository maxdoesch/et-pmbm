#pragma once

#include "tracker/KinematicModel.h"
#include "tracker/Detection.h"
#include "validation/ValidationModel.h"

namespace tracker
{
    class ExtentModel
    {
        public:
            virtual ~ExtentModel() {};
            virtual void predict(double ts) = 0;
            virtual double update(Cluster const& detection) = 0;
            virtual validation::ExtentModel* getExtentValidationModel() const = 0;
            virtual validation::KinematicModel* getKinematicValidationModel() const = 0;
            virtual ExtentModel* copy() const = 0;
    };

    class GIW : public ExtentModel
    {
        public:
            GIW(KinematicModel* k_model);
            GIW(GIW const* e_model);
            ~GIW();
            void predict(double ts) override;
            double update(Cluster const& detection) override;
            validation::ExtentModel* getExtentValidationModel() const override;
            validation::KinematicModel* getKinematicValidationModel() const override;
            ExtentModel* copy() const override;

        private:
            double _v;
            Eigen::Matrix2d _V;

            double const _tau = 1;
            int const _dof = 2;

            KinematicModel* _k_model;
    };  

    class RateModel
    {
        public:
            RateModel();
            RateModel(double alpha, double beta);
            RateModel(RateModel const& r_model);
            void predict();
            double update(Cluster const& detection);
            double getAlpha();
            double getBeta();
            double getRate();
            RateModel operator=(RateModel const& r_model);
            validation::RateModel* getRateValidationModel() const;

        private:
            double _alpha = 1;
            double _beta = 1;

            double const _eta = 1.25; //Estimation and Maintenance of Measurement Rates for Multiple Extended Target Tracking p.5

    };  
}