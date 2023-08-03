#pragma once

#include "validation/ValidationModel.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

#include <Eigen/Dense>

namespace tracker
{
    class Cluster
    {
        public:
            explicit Cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements);
            
            void computeMeanCov();
            int size() const;
            Eigen::Vector2d const& mean() const;
            Eigen::Matrix2d const& covariance() const;

            validation::ValidationModel* getValidationModel(cv::Scalar const& color) const;

        private:
            pcl::PointCloud<pcl::PointXYZ>::Ptr _measurements;

            Eigen::Vector2d _mean;
            Eigen::Matrix2d _covariance;
    };

    class Partition
    {
        public:
            std::vector<Cluster> detections;

            void getValidationModels(std::vector<validation::ValidationModel*>& models) const;
    };
}

