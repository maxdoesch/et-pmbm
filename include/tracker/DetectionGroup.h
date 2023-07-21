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
            DetectionGroup(std::vector<Cluster*> const& detections, std::vector<Bernoulli*> const& bernoullis, PPP const* ppp);
            ~DetectionGroup();
            void createCostMatrix();
            void solve(std::vector<std::vector<Bernoulli*>*>& bernoulli_hypotheses, std::vector<double>& hypotheses_likelihoods);
            void print();

        private:
            std::size_t const _d_size;
            std::size_t const _b_size;

            int const _m_assignments = 1;

            std::vector<Cluster*> const _detections;
            std::vector<Bernoulli*> const _bernoullis;
            PPP const* _ppp;

            MurtyMiller<double>::WeightMatrix _costMatrix;
            std::vector<std::vector<Bernoulli*>*> _bernoulli_matrix;

            std::vector<MurtyMiller<double>::Edges> _assignment_hypotheses;

            std::set<int> _bernoulli_idx;
    };

}