#pragma once

#include <random>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "validation/ValidationModel.h"
#include "simulator/Target.h"

namespace simulator
{
    class Simulator
    {
        public:
            Simulator(double time_step);
            void addTarget(Target* target);
            void addNRandomTargets(int n);
            void step(pcl::PointCloud<pcl::PointXYZ>::Ptr measurements);
            void getValidationModels(std::vector<validation::ValidationModel*>& models);
        
        private:
            double _time = 0;
            std::vector<Target*> _targets;

            double const _time_step;

            double const sim_area_x = 10;
            double const sim_area_y = 5;

            std::random_device _rd;
            std::mt19937 _gen;
    };
}
