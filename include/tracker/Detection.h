#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

#include <Eigen/Dense>

namespace tracker
{
    class Cluster
    {
        public:
            Cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements);

            void add(pcl::PointXYZ& point);
            void add(Eigen::Vector2d& point);
            void computeMeanCov();
            int size() const;
            Eigen::Vector2d const& mean() const;
            Eigen::Matrix2d const& covariance() const;

        private:
            pcl::PointCloud<pcl::PointXYZ>::Ptr _measurements;

            Eigen::Vector2d _mean;
            Eigen::Matrix2d _covariance;
    };
}

