cmake_minimum_required(VERSION 3.20)

project(lstm_c LANGUAGES C CUDA)

find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(lstm lstm.cu)