#include "tracker/Hypothesis.h"

using namespace tracker;

Hypotheses::Hypotheses(double prev_weight, std::vector<Group> const& groups) :
    _prev_weight{prev_weight}
{
    _solve(groups);
}

void Hypotheses::_solve(std::vector<Group> const& groups)
{
    _per_group_hypotheses.reserve(groups.size());

    for(auto const& group : groups)
    {
        MultiBernoulliMixture per_group_hypothesis;
        group.solve(per_group_hypothesis);
        _per_group_hypotheses.push_back(per_group_hypothesis);
    }
}

MultiBernoulliMixture Hypotheses::getMostLikelyHypotheses(int x) const
{
    int max_components = 0;
    for(auto const& per_group_hypothesis : _per_group_hypotheses)
    {
        if(max_components < per_group_hypothesis.size())
            max_components = per_group_hypothesis.size();
    }

    int n = (x < max_components) ? x : max_components;

    
    std::vector<std::vector<MultiBernoulli>> per_group_posterior_mb;
    for(auto const& per_group_hypothesis : _per_group_hypotheses)
    {
        //most_likely_hypotheses.merge(_prev_weight, per_group_hypothesis.selectMostLikely(n));
        per_group_posterior_mb.push_back(per_group_hypothesis.selectMostLikely(n));
    }

    MultiBernoulliMixture most_likely_hypotheses(_prev_weight, per_group_posterior_mb, n);

    return most_likely_hypotheses;
}