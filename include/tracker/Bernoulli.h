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
            void print() const;

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

            void join(MultiBernoulli const& bernoullis);
            std::vector<Bernoulli> const& getBernoullis() const;
            double getWeight() const;
            void setWeight(double weight);

            void getValidationModels(std::vector<validation::ValidationModel*>& models) const;
            void print() const;

            bool operator<(MultiBernoulli const& other) const;

        private:
            std::vector<Bernoulli> _bernoullis;
            double _weight = 0;
    };

    class MultiBernoulliMixture
    {
        public:
            void predict(double ts);
            void prune(double threshold);
            void capping(int N);
            void recycle(double threshold);
            std::vector<Bernoulli> estimate(double threshold);
            void print() const;

            void merge(double prev_weight, std::vector<MultiBernoulli> const& multi_bernoullis);
            void normalize();
            void add(MultiBernoulli const& multi_bernoulli);
            void add(MultiBernoulliMixture const& multi_bernoulli_mixture);
            void clear();
            int size() const;
            std::vector<MultiBernoulli> selectMostLikely(int x) const;
            std::vector<MultiBernoulli> const& getMultiBernoullis();
            MultiBernoulli& operator[](int idx);
            void operator=(MultiBernoulliMixture const& multi_bernoulli_mixture);

        private:
            std::vector<MultiBernoulli> _multi_bernoullis;
    };

} // namespace tracker
