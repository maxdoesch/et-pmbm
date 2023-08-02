#pragma once

#include <limits>

#include "tracker/Group.h"

namespace tracker
{
    class Hypothesis
    {
        public:
            Hypothesis(double prev_weight, std::vector<Group> groups);
            void solve();
            MultiBernoulliMixture getMostLikelyMixture(int x);

        private:
            double _prev_weight = 0;
            std::vector<Group> _groups;
            std::vector<MultiBernoulliMixture> _per_group_hypotheses;
    };
}
    