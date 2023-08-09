#include "tracker/Preprocessing.h"
#include "tracker/constants.h"

#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

using namespace tracker;

PartitionExtractor::PartitionExtractor(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements) : _measurements{measurements}, _two_dim_measurements{new pcl::PointCloud<pcl::PointXYZ>}
{
    for(auto const& point : _measurements->points)
    {
        pcl::PointXYZ point2d;
        point2d.x = point.x;
        point2d.y = point.y;
        point2d.z = 0;
        _two_dim_measurements->push_back(point2d);
    }
}

void PartitionExtractor::getPartitionedParents(std::vector<PartitionedParent>& partitioned_parents)
{
    Partition parent_detections(_measurements, _two_dim_measurements, parent_detection_radius);

    partitioned_parents.reserve(parent_detections.detections.size());

    for(auto const& parent_detection : parent_detections.detections)
    {
        PartitionedParent partitioned_parent(parent_detection);
        partitioned_parents.push_back(partitioned_parent);
    }
}

GroupExtractor::GroupExtractor(std::vector<PartitionedParent> const& paritioned_parents, std::vector<Bernoulli> const& bernoullis, PPP const& ppp) : _ppp{ppp}
{
    for(auto const& partitioned_parent : paritioned_parents)
    {
        _partitioned_parent_associations.push_back(std::make_pair(partitioned_parent, nullptr));
    }

    for(auto const& bernoulli : bernoullis)
    {
        _bernoulli_associations.push_back(std::make_pair(bernoulli, nullptr));
    }
}
/*
void GroupExtractor::extractGroups(std::vector<Group>& groups)
{
    int group_ctr = 0;

    if(_partitioned_parent_associations.size() == 1)
        std::cout << "";

    for(auto& partitioned_parent_association : _partitioned_parent_associations)
    {
        PartitionedParent const& detection = partitioned_parent_association.first;
        int& detection_id = partitioned_parent_association.second; 

        for(auto& bernoulli_association : _bernoulli_associations)
        {
            Bernoulli const& bernoulli = bernoulli_association.first;
            int& bernoulli_id = bernoulli_association.second;

            std::cout << bernoulli.squared_distance(detection.parent) << std::endl;

            if(bernoulli.squared_distance(detection.parent) < gating_threshold)
            {
                if(detection_id == -1)
                {
                    if(bernoulli_id == -1)
                    {
                        Group group(_ppp);
                        group.add(bernoulli);
                        group.id = group_ctr;
                        groups.push_back(group);

                        bernoulli_id = group_ctr;
                        group_ctr++;
                    }

                    int actual_bernoulli_id = groups[bernoulli_id];

                    groups[groups[bernoulli_id].id].add(detection.partitions);
                    detection_id = ;
                }
                else
                {
                    if(bernoulli_id == -1)
                    {
                        groups[detection_id].add(bernoulli);
                        bernoulli_id = detection_id;
                    }
                    else if(detection_id != bernoulli_id)
                    {
                        groups[detection_id].add(groups[bernoulli_id]);
                        groups.erase(groups.begin() + bernoulli_id);
                        bernoulli_id = detection_id;
                    }
                }
            }
        }

        if(detection_id == -1)
        {
            Group group(_ppp);
            group.add(detection.partitions);
            groups.push_back(group);

            group_ctr++;
        }
    }

    for(auto& bernoulli_association : _bernoulli_associations)
    {
        Bernoulli const& bernoulli = bernoulli_association.first;
        int& bernoulli_id = bernoulli_association.second;

        if(bernoulli_id == -1)
        {
            Group group(_ppp);
            group.add(bernoulli);

            groups.push_back(group);
            
            bernoulli_id = group_ctr;
            group_ctr++;
        }
    }
}
*/

void GroupExtractor::extractGroups(std::vector<Group>& groups)
{
    int group_ctr = 0;

    if(_partitioned_parent_associations.size() == 1)
        std::cout << "";

    std::set<Group*> group_ids;

    for(auto& partitioned_parent_association : _partitioned_parent_associations)
    {
        PartitionedParent const& detection = partitioned_parent_association.first;
        Group*& detection_id = partitioned_parent_association.second; 

        for(auto& bernoulli_association : _bernoulli_associations)
        {
            Bernoulli const& bernoulli = bernoulli_association.first;
            Group*& bernoulli_id = bernoulli_association.second;

            if(bernoulli.squared_distance(detection.parent) < gating_threshold)
            {
                if(detection_id == nullptr)
                {
                    if(bernoulli_id == nullptr)
                    {
                        Group* group = new Group(_ppp);
                        group->add(bernoulli);
                        group_ids.insert(group);
                        
                        bernoulli_id = group;
                    }

                    bernoulli_id->add(detection.partitions);
                    detection_id = bernoulli_id;
                }
                else
                {
                    if(bernoulli_id == nullptr)
                    {
                        detection_id->add(bernoulli);
                        bernoulli_id = detection_id;
                    }
                    else if(detection_id != bernoulli_id)
                    {
                        Group* redundant_group_id = bernoulli_id;
                        detection_id->add(*redundant_group_id);
                        for(auto& bernoulli_association_2 : _bernoulli_associations)
                        {
                            if(bernoulli_association_2.second == redundant_group_id)
                            {
                                bernoulli_association_2.second = detection_id;
                            }
                        }
                        for(auto& partitioned_parent_association_2 : _partitioned_parent_associations)
                        {
                            if(partitioned_parent_association_2.second == redundant_group_id)
                            {
                                partitioned_parent_association_2.second = detection_id;
                            }
                        }
                        delete redundant_group_id;
                        group_ids.erase(redundant_group_id);
                    }
                }
            }
        }

        if(detection_id == nullptr)
        {
            Group* group = new Group(_ppp);
            group->add(detection.partitions);
            group_ids.insert(group);

            detection_id = group;
        }
    } 

    for(auto& bernoulli_association : _bernoulli_associations)
    {
        Bernoulli const& bernoulli = bernoulli_association.first;
        Group*& bernoulli_id = bernoulli_association.second;

        if(bernoulli_id == nullptr)
        {
            Group* group = new Group(_ppp);
            group->add(bernoulli);
            group_ids.insert(group);

            bernoulli_id = group;
        }
    }

    for(auto const& group : group_ids)
    {
        groups.push_back(*group);
        delete group;
    }
}