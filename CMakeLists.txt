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
find_package(Boost REQUIRED COMPONENTS thread)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${GNUPLOT_INCLUDE_DIR}
    ${MURTY_INCLUDE_DIR}
    include
)

file(GLOB SRCS "src/**/*.cpp")

# Get all source files in this directory
file(GLOB EXEC "test/*.cpp")

foreach(test_source ${EXEC})
    # Extract the file name without extension
    get_filename_component(test_name ${test_source} NAME_WE)
    
    # Add test executable for each source file
    add_executable(${test_name} ${test_source} ${SRCS})
    
    # Link libraries
    target_link_libraries(${test_name}     
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${Boost_LIBRARIES}
        Boost::thread
        Eigen3::Eigen
    )
endforeach()

add_library(et_pmbm_lib ${SRCS})

target_link_libraries(et_pmbm_lib    
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    Boost::thread
    Eigen3::Eigen
)