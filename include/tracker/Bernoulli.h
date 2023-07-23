#pragma once

#include <vector>
#include <limits>

#include "tracker/ExtentModel.h"
#include "validation/ValidationModel.h"

namespace tracker
{
    class Bernoulli
    {
        public:
            explicit Bernoulli(ExtentModel* e_model);
            Bernoulli(Bernoulli const& bernoulli);
            explicit Bernoulli(double p_existence, ExtentModel* e_model, RateModel const& r_model);
            ~Bernoulli();
            void operator=(Bernoulli const& bernoulli);

            void predict(double ts);
            double missed_detection_likelihood() const;
            double detection_likelihood(Cluster const& detection);
            void update_missed_detection();
            double get_pExistence() const;
            validation::ValidationModel* getValidationModel() const;

        private:
            ExtentModel* _e_model;
            RateModel _r_model;
            double _p_existence = 1;
    };

    class MultiBernoulli
    {
        public:
            MultiBernoulli() {};
            MultiBernoulli(MultiBernoulli const& multi_bernoulli);
            explicit MultiBernoulli(std::vector<Bernoulli> const& bernoullis, double weight);
            void operator=(MultiBernoulli const& multi_bernoulli);

            void predict(double ts);
            void prune(double threshold);

            std::vector<Bernoulli> const& getBernoullis() const;
            double getWeight() const;

            void getValidationModels(std::vector<validation::ValidationModel*>& models) const;

        private:
            std::vector<Bernoulli> _bernoullis;
            double _weight = -std::numeric_limits<double>::infinity();
    };

    class MultiBernoulliMixture
    {
        public:
            void add(MultiBernoulli const& multi_bernoulli);
            MultiBernoulli& operator[](int idx);

        private:
            std::vector<MultiBernoulli> _multi_bernoulli;
    };

} // namespace tracker
