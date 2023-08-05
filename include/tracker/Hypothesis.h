#pragma once

#include <limits>

#include "tracker/Group.h"

namespace tracker
{
    class Hypotheses
    {
        public:
            explicit Hypotheses(double prev_weight, std::vector<Group> const& groups);
            Hypotheses(Hypotheses const& hypotheses) = delete;
            Hypotheses& operator=(Hypotheses const& hypotheses) = delete;
            MultiBernoulliMixture getMostLikelyHypotheses(int x) const;

        private:
            void _solve(std::vector<Group> const& groups);

            double _prev_weight = 0;
            std::vector<MultiBernoulliMixture> _per_group_hypotheses;
    };
}
    