#pragma once

#include <vector>

#include "tracker/Bernoulli.h"
#include "tracker/PoissonPointProcess.h"
#include "tracker/Detection.h"

#include "Miller.h"

namespace tracker
{
    class DetectionGroup
    {
        public:
            explicit DetectionGroup(std::vector<Cluster> const& detections, std::vector<Bernoulli> const& prior_hypothesis, PPP const& ppp);
            DetectionGroup(DetectionGroup const& detection_group) = delete;
            DetectionGroup& operator=(DetectionGroup const& detection_group) = delete;
            ~DetectionGroup();
            
            void solve(MultiBernoulliMixture& detection_hypotheses);
            void print();

        private:
            void _createCostMatrix(std::vector<Cluster> const& detections, PPP const& ppp);
            std::size_t const _d_size;
            std::size_t const _b_size;

            int const _m_assignments = 3;

            MurtyMiller<double>::WeightMatrix _cost_matrix;
            std::vector<std::vector<Bernoulli>> _updated_bernoulli_matrix;
            std::vector<Bernoulli> const& _prior_hypothesis;

            std::vector<MurtyMiller<double>::Edges> _assignment_hypotheses;
    };

}