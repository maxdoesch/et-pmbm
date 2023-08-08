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
            virtual double squared_distance(Cluster const& detection) const = 0;
            virtual validation::ExtentModel* getExtentValidationModel() const = 0;
            virtual validation::KinematicModel* getKinematicValidationModel() const = 0;
            virtual ExtentModel* copy() const = 0;
    };

    template<typename KinematicTemplate> class GIW : public ExtentModel
    {
        public:
            GIW();
            explicit GIW(GIW const* e_model);
            explicit GIW(double const weights[], GIW const e_models[], int components);
            explicit GIW(std::vector<double> const& weights, std::vector<GIW> const& e_models);
            explicit GIW(Eigen::Vector4d const& m, Eigen::Matrix4d const& P, Eigen::Matrix2d const& V);
            ~GIW();
            void operator=(GIW const& e_model);

            void predict(double ts) override;
            double update(Cluster const& detection) override;
            double squared_distance(Cluster const& detection) const override;
            validation::ExtentModel* getExtentValidationModel() const override;
            validation::KinematicModel* getKinematicValidationModel() const override;
            ExtentModel* copy() const override;
            
        private:
            void _merge(GIW& e_model, double const weights[], GIW const e_models[], int components);
            void _merge(std::vector<double> const& weights, std::vector<GIW> const& e_models);
            void _selectMostLikely(GIW& e_model, double const weights[], GIW const e_models[], int components);
            void _selectMostLikely(std::vector<double> const& weights, std::vector<GIW> const& e_models);

            double _v;
            Eigen::Matrix2d _V;
            Eigen::Matrix2d _X_hat;

            double const _tau = 2;
            int const _dof = 2;

            KinematicTemplate _k_model;
    };  

    class RateModel
    {
        public:
            RateModel();
            RateModel(RateModel const& r_model);
            explicit RateModel(double alpha, double beta);
            explicit RateModel(double const weights[], RateModel const r_models[], int components);
            explicit RateModel(std::vector<double> const& weights, std::vector<RateModel> const& r_models);
            void operator=(RateModel const& r_model);

            void predict();
            double update(Cluster const& detection);
            double getAlpha() const;
            double getBeta() const;
            double getRate() const;
            validation::RateModel* getRateValidationModel() const;

        private:
            void _merge(RateModel& r_model_m, double const weights[], RateModel const r_models[], int const& components) const;
            void _merge(std::vector<double> const& weights, std::vector<RateModel> const& r_models);

            double _alpha = 1;
            double _beta = 1;

            double const _eta = 1.25; //Estimation and Maintenance of Measurement Rates for Multiple Extended Target Tracking p.5

    };  
}