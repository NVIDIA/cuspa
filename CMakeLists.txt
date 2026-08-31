cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(cuspa LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # `native` = whatever the build machine has; wheel builds should override
  # this with the desired deployment architecture set, e.g.:
  #   cmake -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;90-real;100-real;120"
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_subdirectory(cpp)
