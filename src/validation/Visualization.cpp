#include "validation/Visualization.h"

using namespace validation;

Visualization::Visualization(double time_step, double time) : 
    _time_step_ms{(int) (1000 * time_step)}, _time_ms{(int) (1000 * time)}, _image{img_size_y, img_size_x, CV_8UC3, cv::Scalar(0,0,0)}, _plot{img_size_y, img_size_x, CV_8UC3, cv::Scalar(255,255,255)}
{
    cv::namedWindow(_window_name, cv::WINDOW_AUTOSIZE);
    outputVideo.open(_vid_file_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 1 / time_step, _image.size());
    _plot = cv::imread(_plt_background_file_name);
}

Visualization::~Visualization()
{
    cv::destroyWindow(_window_name);
    outputVideo.release();
    if(!_plot.empty())
        cv::imwrite(_plt_file_name, _plot);
}

bool Visualization::draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements)
{
    _image.setTo(cv::Scalar(255, 255, 255));

    _draw(_image, measurements);

    cv::imshow(_window_name, _image);

    if(cv::waitKey(_time_step_ms) >= 0)
        return false;

    return true;
}

bool Visualization::draw(std::vector<ValidationModel*> const& models)
{
    _image.setTo(cv::Scalar(255, 255, 255));

    _draw(_image, models);

    cv::imshow(_window_name, _image);

    if(cv::waitKey(_time_step_ms) >= 0)
        return false;

    return true; 
}

bool Visualization::draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements, std::vector<ValidationModel*> const& models)
{
    _image.setTo(cv::Scalar(255, 255, 255));

    _draw(_image, models);
    _draw(_image, measurements);

    cv::imshow(_window_name, _image);

    if(cv::waitKey(_time_step_ms) >= 0)
        return false;

    return true;
}

void Visualization::print(std::vector<ValidationModel*> const& models) const
{
    for(auto const& model : models)
        model->print();
}

void Visualization::plot(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements, std::vector<ValidationModel*> const& models)
{
    int const _total_steps = _time_ms / _time_step_ms;
    int const _plot_model_steps = _total_steps / _plot_model_samples;
    int const _plot_position_steps = _total_steps / _plot_position_samples;
    static int steps = 0;
    static bool halft_time_reached = false; 

    if(!(steps % _plot_model_steps))
    {
        if(steps > _total_steps / 2 && !halft_time_reached)
        {
            _draw(_plot, measurements);
            halft_time_reached = true;
        }
        _draw(_plot, models);
    }
    else if(!(steps % _plot_position_steps))
    {
        for(auto const& model : models)
            model->draw_position(_plot);
    }
        

    steps++;
}

void Visualization::record()
{
    outputVideo << _image;
}

void Visualization::_draw(cv::Mat& image, pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements)
{
    for(auto const& point : *measurements)
    {
        cv::circle(image, cv::Point(point.x * p2co + img_size_x / 2, - point.y * p2co + img_size_y / 2), point_size, cv::Scalar(0, 0, 0), cv::FILLED);
    }
}

void Visualization::_draw(cv::Mat& image, std::vector<ValidationModel*> const& models)
{
    for(auto const& model : models)
    {
        model->draw(image);
    }
}