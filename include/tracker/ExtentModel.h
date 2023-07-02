#pragma once

#include "tracker/KinematicModel.h"
#include "tracker/Detection.h"
#include "validation/ValidationModel.h"

namespace tracker
{
    class ExtentModel
    {
        public:
            virtual void predict(double ts) = 0;
            virtual double update(Cluster const& detection) = 0;
            virtual validation::ValidationModel* getValidationModel() = 0;
    };

    class GGIW : public ExtentModel
    {
        public:
            GGIW(KinematicModel* k_model);
            ~GGIW();
            void predict(double ts) override;
            double update(Cluster const& detection) override;
            validation::ValidationModel* getValidationModel();

        private:
            double _alpha = 0;
            double _beta = 0;
            double _v;
            Eigen::Matrix2d _V;

            double const _eta = 1.25; //Estimation and Maintenance of Measurement Rates for Multiple Extended Target Tracking p.5
            double const _tau = 1;
            int const _dof = 2;

            KinematicModel* _k_model;
    };  
}