#pragma once

#include <vector>

#include "tracker/ExtentModel.h"
#include "validation/ValidationModel.h"

namespace tracker
{
    class Bernoulli
    {
        public:
            Bernoulli(ExtentModel* e_model);
            Bernoulli(Bernoulli const* bernoulli);
            Bernoulli(double p_existence, ExtentModel* e_model, RateModel const& r_model);
            ~Bernoulli();
            void predict(double ts);
            double missed_detection_likelihood() const;
            double detection_likelihood(Cluster const& detection);
            void update_missed_detection();
            double get_pExistence() const;
            validation::ValidationModel* getValidationModel();

        private:
            ExtentModel* _e_model;
            RateModel _r_model;
            double _p_existence = 1;
    };

    class MultiBernoulli
    {
        public:

        private:
            std::vector<Bernoulli*> _bernoullis;
            double _weight;
    };

    class MultiBernoulliMixture
    {
        public:

        private:
            std::vector<MultiBernoulli*> _multiBernoullis;
    };

} // namespace tracker
