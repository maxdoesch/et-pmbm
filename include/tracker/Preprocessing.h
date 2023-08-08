#pragma once

#include <pcl/point_types.h>
#include <pcl/point_types.h>


#include "tracker/Detection.h"
#include "tracker/Bernoulli.h"
#include "tracker/PoissonPointProcess.h"
#include "tracker/Group.h"

#include <utility>
#include <vector>

namespace tracker
{
    class PartitionExtractor
    {
        public:
            explicit PartitionExtractor(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements);
            PartitionExtractor(PartitionExtractor const& partition_extractor) = delete;
            PartitionExtractor& operator=(PartitionExtractor const& partition_extractor) = delete;

            void getPartitionedParents(std::vector<PartitionedParent>& partitioned_parents);

        private:

            pcl::PointCloud<pcl::PointXYZ>::Ptr _measurements;
            pcl::PointCloud<pcl::PointXYZ>::Ptr _two_dim_measurements;
    };

    class GroupExtractor
    {
        public:
            explicit GroupExtractor(std::vector<PartitionedParent> const& paritioned_parents, std::vector<Bernoulli> const& bernoullis, PPP const& ppp);
            GroupExtractor(GroupExtractor const& group_extractor) = delete;
            GroupExtractor& operator=(GroupExtractor const& group_extractor) = delete;

            void extractGroups(std::vector<Group>& groups);

        private:
            PPP const& _ppp;

            std::vector<std::pair<PartitionedParent, Group*>> _partitioned_parent_associations;
            std::vector<std::pair<Bernoulli, Group*>> _bernoulli_associations;
    };
}

