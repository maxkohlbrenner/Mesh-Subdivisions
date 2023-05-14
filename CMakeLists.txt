cmake_minimum_required(VERSION 3.0.0)
project(subdivisions VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 11)

# Add Include Path
include_directories(
	head
	include
)

include(FetchContent)
FetchContent_Declare(
  polyscope
  GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(polyscope)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)
set(EIGEN_BUILD_DOC OFF)
# note: To disable eigen tests,
# you should put this code in a add_subdirectory to avoid to change
# BUILD_TESTING for your own project too since variables are directory
# scoped
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set( OFF)
FetchContent_MakeAvailable(Eigen)

FetchContent_Declare(
  libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG v2.4.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(libigl)

# find source codes
# Add source files
file(GLOB_RECURSE SOURCE_FILES
    src/*.cpp
    head/*.h
    src/**/*.cpp
    head/**/*.h
)
if(WIN32)
add_definitions(-D_AMD64_)
endif()
add_executable(subdivisions ${SOURCE_FILES})
target_link_libraries(subdivisions igl::core polyscope Eigen3::Eigen)
