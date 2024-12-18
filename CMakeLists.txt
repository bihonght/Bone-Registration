# Minimum CMake version required
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

# Project name and build type
PROJECT(STL_Loader)
SET(CMAKE_BUILD_TYPE RelWithDebInfo)

# Enable C++17 standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

# Eigen Library
SET(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third/eigen/")
MESSAGE(STATUS "Using Eigen from: ${EIGEN_INCLUDE_DIR}")
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# Add the source files
SET(SOURCES
    src/STLReader.cpp           # Your example main file
    include/MeshLoader/meshSTLLoader.h # STL loader header
)

# Add the executable
ADD_EXECUTABLE(stl_loader ${SOURCES})

# Link libraries (none required in this example since we're only using Eigen headers)
TARGET_LINK_LIBRARIES(stl_loader PRIVATE)

# Specify include directories for the project
TARGET_INCLUDE_DIRECTORIES(stl_loader PRIVATE include/)
