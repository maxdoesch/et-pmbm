#pragma once

namespace tracker
{
    //monte carlo scenario 1
    double constexpr p_survival = 0.99;
    double constexpr p_detection = 0.99;
    double constexpr clutter_rate = 0.001;
    double constexpr parent_detection_radius = 3;
    double constexpr min_partition_radius = 0.5;
    double constexpr max_partition_radius = 2;
    double constexpr partition_radius_step = 0.25;
    int constexpr min_cluster_size = 5;
    double constexpr gating_threshold = 9;
    double constexpr field_of_view_x = 50;
    double constexpr field_of_view_y = 30;
}
