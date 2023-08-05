#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


#include "tracker/Detection.h"


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
}

