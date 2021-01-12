cmake_minimum_required(VERSION 2.8)

project(obfuscation)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

find_package(CUDA REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS "-lineinfo;-I${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(obfuscation  src/Reduce.cu
                            src/CpuTimer.cpp
                            )
target_include_directories(reduction PUBLIC include)

cuda_add_library(reduction_cuda   src/Reduce.cu)

target_link_libraries(reduction PUBLIC reduction_cuda)
