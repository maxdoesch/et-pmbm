#pragma once

#include "tracker/PoissonPointProcess.h"
#include "tracker/Bernoulli.h"
#include "tracker/Detection.h"

namespace tracker
{
    class Group
    {
        public:
            explicit Group(PPP const& ppp);
            explicit Group(std::vector<Partition> const& partitions, std::vector<Bernoulli> const& bernoullis, PPP const& ppp);
            Group(Group const& group);
            Group& operator=(Group const& group);

            void add(Bernoulli const& bernoulli);
            void add(std::vector<Partition> const& partitions);
            void add(Group const& group);
            void solve(MultiBernoulliMixture& group_hypotheses) const;
            
        private:
            std::vector<Partition> _partitions;
            std::vector<Bernoulli> _bernoullis;
            PPP _ppp;
    };
}