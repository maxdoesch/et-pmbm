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