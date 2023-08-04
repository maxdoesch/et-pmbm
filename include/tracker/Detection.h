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
            bool operator==(Partition const& other) const
            {
                bool same = true;

                if(detections.size() != other.detections.size())
                {
                    same = false;
                }
                else
                {
                    for(auto const& detection : detections)
                    {
                        bool found_detection = false;
                        for(auto const& other_detection : other.detections)
                        {
                            if(detection.mean().isApprox(detection.mean()) && detection.covariance().isApprox(other_detection.covariance()))
                            {
                                found_detection = true;
                                break;
                            }
                        }

                        if(!found_detection)
                        {
                            same = false;
                            break;
                        }
                    }
                }

                return same;
            }
    };
}

