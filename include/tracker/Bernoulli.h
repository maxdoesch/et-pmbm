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
            ~Bernoulli();
            void predict(double ts);
            double likelihood();
            double likelihood(Cluster const& detection, Bernoulli*& bernoulli);
            void update_misdetection(Bernoulli*& bernoulli);
            validation::ValidationModel* getValidationModel();

        private:
            ExtentModel* _e_model;
            double _p_existence = 1;

            double const _p_survival = 0.99;
            double const _p_detection = 0.9;
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
