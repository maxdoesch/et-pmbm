#include "validation/Visualization.h"

using namespace validation;

Visualization::Visualization(double time_steps) : _time_steps_ms{(int) (1000 * time_steps)}, _image{img_size_y, img_size_x, CV_8UC3, cv::Scalar(0,0,0)}
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

    if(cv::waitKey(_time_steps_ms) >= 0)
        return false;

    return true;
}

bool Visualization::draw(std::vector<ValidationModel*> const& models)
{
    _image.setTo(cv::Scalar(255, 255, 255));

    _draw(models);

    cv::imshow(_window_name, _image);

    if(cv::waitKey(_time_steps_ms) >= 0)
        return false;

    return true; 
}

bool Visualization::draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements, std::vector<ValidationModel*> const& models)
{
    _image.setTo(cv::Scalar(255, 255, 255));

    _draw(models);
    _draw(measurements);

    cv::imshow(_window_name, _image);

    if(cv::waitKey(_time_steps_ms) >= 0)
        return false;

    return true;
}

void Visualization::print(std::vector<ValidationModel*> const& models) const
{
    for(auto const& model : models)
        model->print();
}

void Visualization::_draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements)
{
    for(auto const& point : *measurements)
    {
        cv::circle(_image, cv::Point(point.x * p2co + img_size_x / 2, - point.y * p2co + img_size_y / 2), point_size, cv::Scalar(0, 0, 0), cv::FILLED);
    }
}

void Visualization::_draw(std::vector<ValidationModel*> const& models)
{
    for(auto const& model : models)
    {
        model->draw(_image);
    }
}