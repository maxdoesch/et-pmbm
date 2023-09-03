#pragma once

namespace validation
{
    //monte carlo scenario 1
    int constexpr point_size = 2;
    int constexpr dot_size = 8;
    double constexpr coordinate_size_x = 20;
    double constexpr coordinate_size_y = 10;
    int constexpr img_size_x = 2000;
    int constexpr stroke_size = 2;

    double constexpr p2co = img_size_x / (2 * coordinate_size_x);
    int constexpr img_size_y = p2co * (2 * coordinate_size_y);
}