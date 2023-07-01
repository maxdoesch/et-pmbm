#include "validation/Visualization.h"

using namespace validation;

Visualization::Visualization() : _image{_parameters._img_size_y, _parameters._img_size_x, CV_8UC3, cv::Scalar(0,0,0)}
{
    cv::namedWindow(_window_name, cv::WINDOW_AUTOSIZE);
}

Visualization::~Visualization()
{
    cv::destroyWindow(_window_name);
}

bool Visualization::draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements)
{
    _image.setTo(cv::Scalar(255, 255, 255));

    _draw(measurements);

    cv::imshow(_window_name, _image);

    if(cv::waitKey(500) >= 0)
        return false;

    return true;
}

bool Visualization::draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements, std::vector<ValidationModel*> const& models)
{
    _image.setTo(cv::Scalar(255, 255, 255));

    _draw(models);
    _draw(measurements);

    cv::imshow(_window_name, _image);

    if(cv::waitKey(500) >= 0)
        return false;

    return true;
}

void Visualization::_draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements)
{
    for(pcl::PointXYZ& point : *measurements)
    {
        cv::circle(_image, cv::Point(point.x * _parameters._p2co + _parameters._img_size_x / 2, - point.y * _parameters._p2co + _parameters._img_size_y / 2), _parameters._point_size, cv::Scalar(0, 0, 0), cv::FILLED);
    }
}

void Visualization::_draw(std::vector<ValidationModel*> const& models)
{
    for(ValidationModel* model : models)
    {
        model->draw(_image, _parameters);
    }
}