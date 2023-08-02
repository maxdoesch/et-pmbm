cmake_minimum_required(VERSION 3.0)
project(et_pmbm)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GNUPLOT_INCLUDE_DIR "lib/gnuplot-iostream")
set(MURTY_INCLUDE_DIR "lib/MurtyAlgorithm")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PCL 1.9 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${GNUPLOT_INCLUDE_DIR}
    ${MURTY_INCLUDE_DIR}
)

include_directories(include)

# Link directories
link_directories(
    ${PCL_LIBRARY_DIRS}
)

add_executable(et_pmbm src/main.cpp src/simulator/Target.cpp src/simulator/Simulator.cpp src/validation/Visualization.cpp src/validation/ValidationModel.cpp src/tracker/utils.cpp src/tracker/KinematicModel.cpp src/tracker/Detection.cpp src/tracker/ExtentModel.cpp)
add_executable(test_distribution_merging src/test_distribution_merging.cpp src/tracker/utils.cpp)
add_executable(test_bernoulli src/test_bernoulli.cpp src/simulator/Target.cpp src/simulator/Simulator.cpp src/validation/Visualization.cpp src/validation/ValidationModel.cpp src/tracker/utils.cpp src/tracker/KinematicModel.cpp src/tracker/Detection.cpp src/tracker/ExtentModel.cpp src/tracker/Bernoulli.cpp)
add_executable(test_ppp src/test_ppp.cpp src/simulator/Target.cpp src/simulator/Simulator.cpp src/validation/Visualization.cpp src/validation/ValidationModel.cpp src/tracker/utils.cpp src/tracker/KinematicModel.cpp src/tracker/Detection.cpp src/tracker/ExtentModel.cpp src/tracker/Bernoulli.cpp src/tracker/PoissonPointProcess.cpp)
add_executable(test_detection_group src/test_detection_group.cpp src/simulator/Target.cpp src/simulator/Simulator.cpp src/validation/Visualization.cpp src/validation/ValidationModel.cpp src/tracker/utils.cpp src/tracker/KinematicModel.cpp src/tracker/Detection.cpp src/tracker/ExtentModel.cpp src/tracker/Bernoulli.cpp src/tracker/PoissonPointProcess.cpp src/tracker/DetectionGroup.cpp)
add_executable(test_multi_bernoulli_mixture src/test_multi_bernoulli_mixture.cpp src/simulator/Target.cpp src/simulator/Simulator.cpp src/validation/Visualization.cpp src/validation/ValidationModel.cpp src/tracker/utils.cpp src/tracker/KinematicModel.cpp src/tracker/Detection.cpp src/tracker/ExtentModel.cpp src/tracker/Bernoulli.cpp src/tracker/PoissonPointProcess.cpp src/tracker/DetectionGroup.cpp src/tracker/Group.cpp src/tracker/Hypothesis.cpp)



# Link libraries
target_link_libraries(et_pmbm
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)

target_link_libraries(test_distribution_merging
    -lboost_iostreams 
)

target_link_libraries(test_bernoulli
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)

target_link_libraries(test_ppp
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)

target_link_libraries(test_detection_group
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)

target_link_libraries(test_multi_bernoulli_mixture
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)