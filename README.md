# ET-GGIW-PMBM
This repository contains the C++ Implementation of the Extended Target Gamma Gaussian Inverse Wishart Poisson Multi-Bernoulli Mixture Filter for a Bachelor Thesis at the University of Applied Sciences Nuremberg:

`Design and Implementation of a Lidar-Based Multiple Object Tracking System for Autonomous Shunting Locomotives` 

![](images/tracker_demo.gif)

# Requirements
`OpenCV`
`Boost`
`Eigen`


# Build
```sh
git clone --recurse-submodules https://github.com/maxdoesch/et-pmbm.git
mkdir build && cd build && cmake .. && make
```
# Run Example
```sh
cd build
./test_tracker
```
