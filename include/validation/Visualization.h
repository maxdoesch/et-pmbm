#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/videoio.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "validation/ValidationModel.h"
#include "validation/constants.h"


namespace validation
{
    class Visualization
    {
        public:
            Visualization(double time_step, double time);
            ~Visualization();
            bool draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements);
            bool draw(std::vector<ValidationModel*> const& models);
            bool draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements, std::vector<ValidationModel*> const& models);
            void print(std::vector<ValidationModel*> const& models) const;
            void plot(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements, std::vector<ValidationModel*> const& models);
            void record();

        private:
            void _draw(cv::Mat& image, pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements);
            void _draw(cv::Mat& image, std::vector<ValidationModel*> const& models);

            int _time_step_ms = 0;
            int _time_ms = 0;

            cv::Mat _image;
            cv::Mat _plot;
            cv::VideoWriter outputVideo;

            int const _plot_model_samples = 5;
            int const _plot_position_samples = 10;

            std::string const _window_name = "Visualization";
            std::string const _vid_file_name = "tracking.avi";
            std::string const _plt_file_name = "plot.png";
            std::string const _plt_background_file_name = "../images/coordinate_system.png";
    };
}
