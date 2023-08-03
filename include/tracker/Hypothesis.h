#pragma once

#include <limits>

#include "tracker/Group.h"

namespace tracker
{
    class Hypotheses
    {
        public:
            Hypotheses(double prev_weight, std::vector<Group> const& groups);
            MultiBernoulliMixture getMostLikelyHypotheses(int x) const;

        private:
            void _solve(std::vector<Group> const& groups);

            double _prev_weight = 0;
            std::vector<MultiBernoulliMixture> _per_group_hypotheses;
    };
}
    