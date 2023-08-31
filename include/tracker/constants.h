#pragma once

namespace tracker
{
    double constexpr p_survival = 0.99;
    double constexpr p_detection = 0.99;
    double constexpr clutter_rate = 0.001;
    double constexpr min_likelihood = -500;
    double constexpr max_hypothesis_l_weight_diff = 500;
    constexpr double parent_detection_radius = 3.0;
    constexpr double min_partition_radius = 0.1;
    constexpr double max_partition_radius = 2.0;
    constexpr double partition_radius_step = 0.15;
    constexpr int min_cluster_size = 3;
    constexpr double gating_threshold = 9;
}
