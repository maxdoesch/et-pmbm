#include "tracker/Hypothesis.h"

using namespace tracker;

Hypothesis::Hypothesis(double prev_weight, std::vector<Group> groups) :
    _prev_weight{prev_weight}, _groups{groups}
{

}

void Hypothesis::solve()
{
    _per_group_hypotheses.reserve(_groups.size());

    for(auto const& group : _groups)
    {
        MultiBernoulliMixture per_group_hypothesis = group.solve();
        _per_group_hypotheses.push_back(per_group_hypothesis);
    }
}

MultiBernoulliMixture Hypothesis::getMostLikelyMixture(int x)
{
    int max_components = 0;
    for(auto const& per_group_hypothesis : _per_group_hypotheses)
    {
        if(max_components < per_group_hypothesis.size())
            max_components = per_group_hypothesis.size();
    }

    int n = (x < max_components) ? x : max_components;

    MultiBernoulliMixture mostLikelyMixture;

    for(auto const& per_group_hypothesis : _per_group_hypotheses)
    {
        mostLikelyMixture.merge(_prev_weight, per_group_hypothesis.selectMostLikely(n));
    }

    return mostLikelyMixture;
}