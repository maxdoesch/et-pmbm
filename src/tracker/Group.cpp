#include "tracker/Group.h"

#include "tracker/DetectionGroup.h"

using namespace tracker;

Group::Group(std::vector<Partition> const& partitions, std::vector<Bernoulli> const& bernoullis, PPP const& ppp) :
    _partitions{partitions}, _bernoullis{bernoullis}, _ppp{ppp}
{

}

MultiBernoulliMixture Group::solve() const
{
    MultiBernoulliMixture multi_bernoulli_mixture;

    for(auto const& partition : _partitions)
    {
        DetectionGroup detection_group(partition.detections, _bernoullis, _ppp);
        detection_group.solve(multi_bernoulli_mixture);
    }

    return multi_bernoulli_mixture;
}
