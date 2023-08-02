#pragma once

#include "tracker/PoissonPointProcess.h"
#include "tracker/Bernoulli.h"
#include "tracker/Detection.h"

namespace tracker
{
    class Group
    {
        public:
            Group(std::vector<Partition> const& partitions, std::vector<Bernoulli> const& bernoullis, PPP const& ppp);
            MultiBernoulliMixture solve() const;
            

        private:
            std::vector<Partition> _partitions;
            std::vector<Bernoulli> _bernoullis;
            PPP const& _ppp;
    };
}