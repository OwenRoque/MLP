#pragma once

#include "core/cuda_utils.cuh"

#include <cstddef>
#include <vector>

/// Contenedor RAII para memoria en GPU (y copia opcional en host).
class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::size_t size, bool allocate_host = false)
        : size_(size), allocate_host_(allocate_host) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&device_data_, size_ * sizeof(float)));
            if (allocate_host_) {
                host_data_.resize(size_);
            }
        }
    }

    ~Tensor() { release(); }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept { move_from(std::move(other)); }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    float* device_data() { return device_data_; }
    const float* device_data() const { return device_data_; }

    float* host_data() { return host_data_.data(); }
    const float* host_data() const { return host_data_.data(); }

    std::size_t size() const { return size_; }
    bool has_host() const { return allocate_host_; }

    void copy_to_device(const float* host_src) {
        CUDA_CHECK(cudaMemcpy(device_data_, host_src, size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void copy_to_device(const std::vector<float>& host_src) {
        copy_to_device(host_src.data());
    }

    void copy_to_host() {
        if (!allocate_host_) {
            host_data_.resize(size_);
            allocate_host_ = true;
        }
        CUDA_CHECK(cudaMemcpy(host_data_.data(), device_data_, size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void zero_device() {
        if (device_data_ != nullptr) {
            CUDA_CHECK(cudaMemset(device_data_, 0, size_ * sizeof(float)));
        }
    }

private:
    float* device_data_ = nullptr;
    std::vector<float> host_data_;
    std::size_t size_ = 0;
    bool allocate_host_ = false;

    void release() {
        if (device_data_ != nullptr) {
            cudaFree(device_data_);
            device_data_ = nullptr;
        }
        host_data_.clear();
        size_ = 0;
        allocate_host_ = false;
    }

    void move_from(Tensor&& other) noexcept {
        device_data_ = other.device_data_;
        host_data_ = std::move(other.host_data_);
        size_ = other.size_;
        allocate_host_ = other.allocate_host_;
        other.device_data_ = nullptr;
        other.size_ = 0;
        other.allocate_host_ = false;
    }
};
