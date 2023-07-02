#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "validation/ValidationModel.h"
#include "validation/settings.h"


namespace validation
{
    class Visualization
    {
        public:
            Visualization(double time_steps);
            ~Visualization();
            bool draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements);
            bool draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements, std::vector<ValidationModel*> const& models);

        private:
            void _draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements);
            void _draw(std::vector<ValidationModel*> const& models);

            int _time_steps_ms = 0;

            Parameters const _parameters;
            cv::Mat _image;

            std::string const _window_name = "Visualization";
    };
}
