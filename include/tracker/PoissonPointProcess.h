#pragma once

#include <vector>

#include "tracker/ExtentModel.h"
#include "validation/ValidationModel.h"


namespace tracker
{
    class Bernoulli;
    class PoissonComponent
    {
        public:
            explicit PoissonComponent(double weight, GIW<ConstantVelocity> const& e_model, RateModel const& r_model);
            explicit PoissonComponent(double l_weight, Bernoulli const& bernoulli);
            PoissonComponent(PoissonComponent const& p_component);
            ~PoissonComponent();
            PoissonComponent& operator=(PoissonComponent const& p_component);

            void predict(double ts);
            void update_missed_detection();
            double detection_likelihood(Cluster const& detection, GIW<ConstantVelocity>& e_model, RateModel& r_model) const;
            double getWeight() const;
            validation::ValidationModel* getValidationModel() const;

            bool operator>(PoissonComponent const& p_component) const;

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
            BirthModel(BirthModel const& birth_model);
            BirthModel& operator=(BirthModel const& birth_model);
            void birth(std::vector<PoissonComponent>& b_components) const;

        private:
            std::vector<PoissonComponent> _birth_components;

            int const _n_components = 4;

            double const _field_of_view_x = 1280. / 50.;
            double const _field_of_view_y = 720. / 50.;
    };
    class PPP
    {
        public:
            PPP();
            ~PPP();
            PPP(PPP const& ppp);
            PPP& operator=(PPP const& ppp);

            void predict(double ts);
            void update_missed_detection();
            Bernoulli detection_likelihood(Cluster const& detection, double& likelihood) const;
            void prune(double threshold);
            void capping(int N);
            void add_component(PoissonComponent const& p_component);
            void getValidationModels(std::vector<validation::ValidationModel*>& models);

        private:
            std::vector<PoissonComponent> _p_components;
            BirthModel _b_model;

            double const _min_likelihood = 0.0001;
    };
}