#pragma once

#include <vector>

#include "tracker/ExtentModel.h"
#include "tracker/Bernoulli.h"
#include "validation/ValidationModel.h"


namespace tracker
{
    class PoissonComponent
    {
        public:
            PoissonComponent(double weight, GIW<ConstantVelocity> const& e_model, RateModel const& r_model);
            PoissonComponent(PoissonComponent const& p_component);
            ~PoissonComponent();
            void predict(double ts);
            void update_missed_detection();
            double detection_likelihood(Cluster const& detection, GIW<ConstantVelocity>& e_model, RateModel& r_model) const;
            double getWeight() const;
            validation::ValidationModel* getValidationModel() const;

        private:
            GIW<ConstantVelocity> _e_model;
            RateModel _r_model;
            double _weight = 1;
    };

    class BirthModel
    {
        public:
            BirthModel();
            ~BirthModel();
            void birth(std::vector<PoissonComponent*>& b_components) const;

        private:
            std::vector<PoissonComponent> _birth_components;

            int const _n_components = 4;

            double const _field_of_view_x = 1280. / 50.;
            double const _field_of_view_y = 720. / 50.;
    };

    class PPP
    {
        public:
            PPP() {};
            ~PPP();
            void predict(double ts);
            void update_missed_detection();
            double detection_likelihood(Cluster const& detection, Bernoulli*& bernoulli) const;
            Bernoulli detection_likelihood(Cluster const& detection, double& likelihood) const;
            void getValidationModels(std::vector<validation::ValidationModel*>& models);

        private:
            std::vector<PoissonComponent*> _p_components;

            BirthModel _b_model;
    };
}