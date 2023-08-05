#include "tracker/Detection.h"
#include "tracker/constants.h"

#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

using namespace tracker;

Cluster::Cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements) :
 _measurements{measurements}, _two_dim_measurements{new pcl::PointCloud<pcl::PointXYZ>}, _mean{Eigen::Vector2d::Zero()}, _covariance{Eigen::Matrix2d::Zero()}
{
    _computeMeanCov();
}

Cluster::Cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr const& measurements, pcl::PointCloud<pcl::PointXYZ>::Ptr const& two_dim_measurements) : 
_measurements{measurements}, _two_dim_measurements{two_dim_measurements}, _mean{Eigen::Vector2d::Zero()}, _covariance{Eigen::Matrix2d::Zero()}
{
    _computeMeanCov();
}

Cluster::Cluster(Cluster const& cluster) : 
_measurements{cluster._measurements}, _two_dim_measurements{cluster._two_dim_measurements}, _mean{cluster._mean}, _covariance{cluster._covariance}
{

}

void Cluster::_computeMeanCov()
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

pcl::PointCloud<pcl::PointXYZ>::Ptr Cluster::measurements() const
{
    return _measurements;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Cluster::two_dim_measurements() const
{
    return _two_dim_measurements;
}


validation::ValidationModel* Cluster::getValidationModel(cv::Scalar const& color) const
{
    Eigen::Matrix2d covariance = 4 * _covariance / _measurements->points.size() + Eigen::Matrix2d::Identity() * 0.0001;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigenSolver(covariance);
    Eigen::Matrix2d eigenVectors = eigenSolver.eigenvectors();
    Eigen::Vector2d eigenValues = eigenSolver.eigenvalues();

    // Sorting eigenvalues in descending order
    if (eigenValues(0) > eigenValues(1))
    {
        std::swap(eigenValues(0), eigenValues(1));
        eigenVectors.col(0).swap(eigenVectors.col(1));
    }

    // Semi-major and semi-minor axes
    double a = std::sqrt(eigenValues(0));
    double b = std::sqrt(eigenValues(1));

    // Rotation angle (in radians)
    double angle = std::atan2(eigenVectors(1, 0), eigenVectors(0, 0));

    Eigen::Matrix<double, 5, 1> state;
    state << _mean, 0, 0, angle;

    validation::Ellipse* ellipse = new validation::Ellipse(a, b);
    validation::ConstantVelocity* cv = new validation::ConstantVelocity(state);
    validation::RateModel* rate_model = new validation::RateModel(_measurements->points.size());

    return new validation::GenericValidationModel(cv, ellipse, rate_model, color);
}

void Partition::getValidationModels(std::vector<validation::ValidationModel*>& models) const
{
    int r = (double)std::rand() / RAND_MAX * 255;
    int g = (double)std::rand() / RAND_MAX * 255;
    int b = (double)std::rand() / RAND_MAX * 255;

    for(auto const& detection : detections)
    {
        models.push_back(detection.getValidationModel(CV_RGB(r, g, b)));
    }
}

Partition::Partition(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements, double radius)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZ>);
    kdTree->setInputCloud(measurements);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> pcl_euclidean_cluster;
    pcl_euclidean_cluster.setClusterTolerance(radius);
    pcl_euclidean_cluster.setMinClusterSize(min_cluster_size);
    pcl_euclidean_cluster.setSearchMethod(kdTree);
    pcl_euclidean_cluster.setInputCloud(measurements);
    pcl_euclidean_cluster.extract(cluster_indices);

    detections.reserve(cluster_indices.size());
    for (pcl::PointIndices const& point_indices : cluster_indices) 
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_measurements(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto & point_idx : point_indices.indices) 
        {
            cluster_measurements->points.push_back(measurements->points[point_idx]);
        }

        cluster_measurements->width = cluster_measurements->points.size();
        cluster_measurements->height = 1;
        cluster_measurements->is_dense = false;

        tracker::Cluster detection(cluster_measurements);

        detections.push_back(detection);
    }
}

Partition::Partition(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements, pcl::PointCloud<pcl::PointXYZ>::Ptr two_dim_measurements, double radius)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZ>);
    kdTree->setInputCloud(two_dim_measurements);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> pcl_euclidean_cluster;
    pcl_euclidean_cluster.setClusterTolerance(radius);
    pcl_euclidean_cluster.setMinClusterSize(min_cluster_size);
    pcl_euclidean_cluster.setSearchMethod(kdTree);
    pcl_euclidean_cluster.setInputCloud(two_dim_measurements);
    pcl_euclidean_cluster.extract(cluster_indices);

    detections.reserve(cluster_indices.size());
    for (pcl::PointIndices const& point_indices : cluster_indices) 
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_measurements(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr two_dim_cluster_measurements(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto & point_idx : point_indices.indices) 
        {
            cluster_measurements->points.push_back(measurements->points[point_idx]);
            two_dim_cluster_measurements->points.push_back(two_dim_measurements->points[point_idx]);
        }

        cluster_measurements->width = cluster_measurements->points.size();
        cluster_measurements->height = 1;
        cluster_measurements->is_dense = false;

        two_dim_cluster_measurements->width = two_dim_cluster_measurements->points.size();
        two_dim_cluster_measurements->height = 1;
        two_dim_cluster_measurements->is_dense = false;

        tracker::Cluster detection(cluster_measurements, two_dim_cluster_measurements);

        detections.push_back(detection);
    }
}

Partition::Partition(Partition const& partition) : detections{partition.detections}
{

}

PartitionedParent::PartitionedParent(Cluster const& parent_cluster) : parent{parent_cluster}
{
    int n_detections = 0;

    for(int i = (max_partition_radius - min_partition_radius) / partition_radius_step; i >= 0; i--)
    {
        tracker::Partition new_partition(parent.measurements(), parent.two_dim_measurements(), i * partition_radius_step + min_partition_radius);

        if(n_detections < new_partition.detections.size())
        {
            n_detections = new_partition.detections.size();

            partitions.push_back(new_partition);
        }
    }
}

PartitionedParent::PartitionedParent(PartitionedParent const& partitioned_parent) : parent{partitioned_parent.parent}, partitions{partitioned_parent.partitions}
{
    
}
