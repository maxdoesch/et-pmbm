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

validation::ValidationModel* Cluster::getValidationModel() const
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

    return new validation::GenericValidationModel(cv, ellipse, rate_model, CV_RGB(255, 255, 0));
}

void Partition::getValidationModels(std::vector<validation::ValidationModel*>& models) const
{
    for(auto const& detection : detections)
    {
        models.push_back(detection.getValidationModel());
    }
}