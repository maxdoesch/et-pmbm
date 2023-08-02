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
            Visualization(double time_steps);
            ~Visualization();
            bool draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements);
            bool draw(std::vector<ValidationModel*> const& models);
            bool draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements, std::vector<ValidationModel*> const& models);
            void print(std::vector<ValidationModel*> const& models) const;
            void record();

        private:
            void _draw(pcl::PointCloud<pcl::PointXYZ>::Ptr const & measurements);
            void _draw(std::vector<ValidationModel*> const& models);

            int _time_steps_ms = 0;

            cv::Mat _image;
            cv::VideoWriter outputVideo;

            std::string const _window_name = "Visualization";
            std::string const _vid_file_name = "tracking.avi";
    };
}
