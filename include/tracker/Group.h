#pragma once

#include "tracker/PoissonPointProcess.h"
#include "tracker/Bernoulli.h"
#include "tracker/Detection.h"

namespace tracker
{
    class Group
    {
        public:
            explicit Group(std::vector<Partition> const& partitions, std::vector<Bernoulli> const& bernoullis, PPP const& ppp);
            void solve(MultiBernoulliMixture& group_hypotheses) const;
            

        private:
            std::vector<Partition> const& _partitions;
            std::vector<Bernoulli> const& _bernoullis;
            PPP const& _ppp;
    };
}