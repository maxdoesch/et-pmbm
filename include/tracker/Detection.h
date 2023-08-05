#pragma once

#include "validation/ValidationModel.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

#include <Eigen/Dense>

#include <vector>

namespace tracker
{
    class Cluster
    {
        public:
            explicit Cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr const& measurements);
            explicit Cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr const& measurements, pcl::PointCloud<pcl::PointXYZ>::Ptr const& two_dim_measurements);
            Cluster(Cluster const& cluster);
            Cluster& operator=(Cluster const& cluster) = delete;
            
            int size() const;
            Eigen::Vector2d const& mean() const;
            Eigen::Matrix2d const& covariance() const;
            pcl::PointCloud<pcl::PointXYZ>::Ptr measurements() const;
            pcl::PointCloud<pcl::PointXYZ>::Ptr two_dim_measurements() const;


            validation::ValidationModel* getValidationModel(cv::Scalar const& color) const;

        private:
            void _computeMeanCov();

            pcl::PointCloud<pcl::PointXYZ>::Ptr _measurements;
            pcl::PointCloud<pcl::PointXYZ>::Ptr _two_dim_measurements;

            Eigen::Vector2d _mean;
            Eigen::Matrix2d _covariance;
    };

    class Partition
    {
        public:
            Partition(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements, double radius);
            Partition(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements, pcl::PointCloud<pcl::PointXYZ>::Ptr two_dim_measurements, double radius);
            Partition(Partition const& partition);
            Partition& operator=(Partition const& partition) = delete;

            void getValidationModels(std::vector<validation::ValidationModel*>& models) const;

            std::vector<Cluster> detections;
    };

    class PartitionedParent
    {
        public:
            explicit PartitionedParent(Cluster const& parent_cluster);
            PartitionedParent(PartitionedParent const& partitioned_parent);
            PartitionedParent& operator=(PartitionedParent const& partitioned_parent) = delete;

            Cluster parent;
            std::vector<Partition> partitions;
    };
}

