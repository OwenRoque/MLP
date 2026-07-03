#include "core/profiling.hpp"

#include <algorithm>

namespace {

std::size_t g_peak_gpu_memory_bytes = 0;

} // namespace

GpuMemoryStats query_gpu_memory() {
    CUDA_CHECK(cudaDeviceSynchronize());

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    GpuMemoryStats stats;
    stats.total_bytes = total_bytes;
    stats.free_bytes = free_bytes;
    stats.used_bytes = total_bytes - free_bytes;
    return stats;
}

void update_peak_gpu_memory(std::size_t& peak_bytes) {
    const GpuMemoryStats stats = query_gpu_memory();
    peak_bytes = std::max(peak_bytes, stats.used_bytes);
    g_peak_gpu_memory_bytes = std::max(g_peak_gpu_memory_bytes, stats.used_bytes);
}

std::size_t peak_gpu_memory_bytes() {
    return g_peak_gpu_memory_bytes;
}

void reset_peak_gpu_memory() {
    g_peak_gpu_memory_bytes = 0;
}

void WallTimer::start() {
    start_ = std::chrono::steady_clock::now();
    running_ = true;
}

double WallTimer::elapsed_ms() const {
    if (!running_) {
        return 0.0;
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
}

double WallTimer::elapsed_sec() const {
    return elapsed_ms() / 1000.0;
}
