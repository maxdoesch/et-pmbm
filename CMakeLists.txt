cmake_minimum_required(VERSION 3.0)
project(et_pmbm)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PCL 1.9 REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

include_directories(include)

# Link directories
link_directories(
    ${PCL_LIBRARY_DIRS}
)

add_executable(et_pmbm src/main.cpp src/simulator/Target.cpp src/validation/Visualization.cpp src/validation/ValidationModel.cpp src/tracker/utils.cpp src/tracker/KinematicModel.cpp src/tracker/Detection.cpp src/tracker/ExtentModel.cpp)

# Link libraries
target_link_libraries(et_pmbm
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)