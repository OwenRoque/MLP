#pragma once

#include "core/cuda_utils.cuh"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

struct GpuMemoryStats {
    std::size_t total_bytes = 0;
    std::size_t free_bytes = 0;
    std::size_t used_bytes = 0;
};

/// Consulta memoria GPU actual (sincroniza el dispositivo antes de medir).
GpuMemoryStats query_gpu_memory();

/// Actualiza el pico de memoria usada si el valor actual es mayor.
void update_peak_gpu_memory(std::size_t& peak_bytes);

/// Pico de memoria GPU usada desde el inicio del programa.
std::size_t peak_gpu_memory_bytes();

/// Reinicia el contador de pico de memoria GPU.
void reset_peak_gpu_memory();

/// Temporizador de pared (CPU) para medir tiempo de ejecución.
class WallTimer {
public:
    void start();
    double elapsed_ms() const;
    double elapsed_sec() const;

private:
    std::chrono::steady_clock::time_point start_{};
    bool running_ = false;
};

inline double bytes_to_mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

inline std::string format_mib(std::size_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << bytes_to_mib(bytes) << " MiB";
    return oss.str();
}

inline std::ostream& operator<<(std::ostream& out, const GpuMemoryStats& stats) {
    out << format_mib(stats.used_bytes) << " / " << format_mib(stats.total_bytes);
    return out;
}
