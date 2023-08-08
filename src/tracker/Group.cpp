#include "tracker/Group.h"

#include "tracker/DetectionGroup.h"

using namespace tracker;

Group::Group(PPP const& ppp) : _ppp{ppp}
{

}

Group::Group(std::vector<Partition> const& partitions, std::vector<Bernoulli> const& bernoullis, PPP const& ppp) :
    _partitions{partitions}, _bernoullis{bernoullis}, _ppp{ppp}
{

}

Group::Group(Group const& group) : _partitions{group._partitions}, _bernoullis{group._bernoullis}, _ppp{group._ppp}
{

}

Group& Group::operator=(Group const& group)
{
    _partitions = group._partitions;
    _bernoullis = group._bernoullis;
    _ppp = group._ppp;

    return *this;
}

void Group::add(Bernoulli const& bernoulli)
{
    _bernoullis.push_back(bernoulli);
}

void Group::add(std::vector<Partition> const& partitions)
{
    if(_partitions.empty())
        _partitions.insert(_partitions.end(), partitions.begin(), partitions.end());
    else
    {
        int i = 0;

        Partition back_partition = _partitions.back();

        for(auto const& partition : partitions)
        {
            if(i >= _partitions.size())
            {
                Partition new_partition(partition);
                new_partition.merge(back_partition);
                _partitions.push_back(new_partition);
            }
            else
                _partitions[i].merge(partition);

            i++;
        }

        for(i = partitions.size(); i < _partitions.size(); i++)
            _partitions[i].merge(partitions.back());
    }
}

void Group::add(Group const& group)
{
    _bernoullis.reserve(_bernoullis.size() + group._bernoullis.size());
    _bernoullis.insert(_bernoullis.end(), group._bernoullis.begin(), group._bernoullis.end());

    add(group._partitions);
}

void Group::solve(MultiBernoulliMixture& group_hypotheses) const
{
    if(_partitions.empty())
    {
        std::vector<Cluster> detections;
        DetectionGroup detection_group(detections, _bernoullis, _ppp);
        detection_group.solve(group_hypotheses);
    }

    for(auto const& partition : _partitions)
    {
        DetectionGroup detection_group(partition.detections, _bernoullis, _ppp);
        detection_group.solve(group_hypotheses);
        detection_group.print();
    }
}
