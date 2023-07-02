#include "tracker/Detection.h"

using namespace tracker;

Cluster::Cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements) : _measurements{measurements}
{

}

void Cluster::computeMeanCov()
{
    Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();

    for(auto point : *_measurements)
    {
        mean[0] += point.x;
        mean[1] += point.y;
    }

    mean /= _measurements->points.size();

    for(auto point : *_measurements)
    {
        Eigen::Vector2d measurement = {point.x, point.y};
        Eigen::Vector2d delta = measurement - mean;
        covariance += delta * delta.transpose();
    }

    _mean = mean;
    _covariance = covariance;
}

int Cluster::size() const
{
    return _measurements->points.size();
}

Eigen::Vector2d const& Cluster::mean() const
{
    return _mean;
}

Eigen::Matrix2d const& Cluster::covariance() const
{
    return _covariance;
}