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
            virtual double getAlpha() = 0;
            virtual double getBeta() = 0;
            virtual void setAlpha(double const& alpha) = 0;
            virtual void setBeta(double const& beta) = 0;
            virtual validation::ValidationModel* getValidationModel() = 0;
            virtual ExtentModel* copy() const = 0;
    };

    class GGIW : public ExtentModel
    {
        public:
            GGIW(KinematicModel* k_model);
            GGIW(GGIW const* e_model);
            ~GGIW();
            void predict(double ts) override;
            double update(Cluster const& detection) override;
            double getAlpha() override;
            double getBeta() override;
            void setAlpha(double const& alpha) override;
            void setBeta(double const& beta) override;
            validation::ValidationModel* getValidationModel();
            ExtentModel* copy() const override;

        private:
            double _alpha = 1;
            double _beta = 1;
            double _v;
            Eigen::Matrix2d _V;

            double const _eta = 1.25; //Estimation and Maintenance of Measurement Rates for Multiple Extended Target Tracking p.5
            double const _tau = 1;
            int const _dof = 2;

            KinematicModel* _k_model;
    };  
}