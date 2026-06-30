#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            " — " + cudaGetErrorString(err));
    }
}

#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)
