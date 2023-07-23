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
            explicit Bernoulli(Bernoulli const* bernoulli);
            Bernoulli(Bernoulli const& bernoulli);
            explicit Bernoulli(double p_existence, ExtentModel* e_model, RateModel const& r_model);

            ~Bernoulli();
            void predict(double ts);
            double missed_detection_likelihood() const;
            double detection_likelihood(Cluster const& detection);
            void update_missed_detection();
            double get_pExistence() const;
            validation::ValidationModel* getValidationModel() const;
            void operator=(Bernoulli const& bernoulli);

        private:
            ExtentModel* _e_model;
            RateModel _r_model;
            double _p_existence = 1;
    };

    class MultiBernoulli
    {
        public:
            MultiBernoulli() {};
            explicit MultiBernoulli(MultiBernoulli const& multi_bernoulli);
            MultiBernoulli(std::vector<Bernoulli> const& bernoullis, double weight);

            void predict(double ts);
            void prune(double threshold);

            std::vector<Bernoulli> const& getBernoullis() const;
            double getWeight() const;

            void getValidationModels(std::vector<validation::ValidationModel*>& models) const;

            void operator=(MultiBernoulli const& multi_bernoulli);

        private:
            std::vector<Bernoulli> _bernoullis;
            double _weight = -std::numeric_limits<double>::infinity();
    };

    class MultiBernoulliMixture
    {
        public:
            void add(MultiBernoulli const& multiBernoulli);
            MultiBernoulli& operator[](int idx);

        private:
            std::vector<MultiBernoulli> _multiBernoullis;
    };

} // namespace tracker
