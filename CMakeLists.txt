cmake_minimum_required(VERSION 3.18)
project(mnist_mlp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(ZLIB REQUIRED)

# Compila para la GPU local si no se especifica arquitectura (CMake >= 3.24).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(SOURCES
    src/main.cpp
    src/core/profiling.cpp
    src/data/mnist_loader.cpp
    src/network/network.cpp
    src/training/trainer.cpp
    src/inference/inference_export.cpp
    src/cuda/linear_layer.cu
    src/cuda/activation_layer.cu
    src/cuda/softmax_cross_entropy.cu
)

add_executable(mnist_mlp ${SOURCES})

target_include_directories(mnist_mlp PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(mnist_mlp PRIVATE CUDA::cudart ZLIB::ZLIB)

target_compile_options(mnist_mlp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)
