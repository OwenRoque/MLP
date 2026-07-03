#pragma once

#include "core/tensor.hpp"
#include "data/mnist_loader.hpp"
#include "network/network.hpp"

#include <string>

struct TrainConfig {
    int epochs = 10;
    int batch_size = 128;
    float learning_rate = 0.1f;
    int log_every = 100; // batches entre logs de entrenamiento
};

struct TrainMetrics {
    float loss = 0.0f;
    float accuracy = 0.0f;
    double elapsed_sec = 0.0;
    std::size_t gpu_memory_used_bytes = 0;
    std::size_t gpu_memory_peak_bytes = 0;
};

class Trainer {
public:
    Trainer(Network& network, const TrainConfig& config);
    ~Trainer();

    TrainMetrics train_epoch(const MNISTDataset& dataset);
    TrainMetrics evaluate(const MNISTDataset& dataset);

    std::size_t peak_gpu_memory_bytes() const { return peak_gpu_memory_bytes_; }

private:
    Network& network_;
    TrainConfig config_;

    Tensor d_batch_input_;
    Tensor d_grad_output_;
    int* d_batch_labels_ = nullptr;
    int max_batch_size_ = 0;
    std::size_t peak_gpu_memory_bytes_ = 0;

    void upload_batch(const MNISTDataset& dataset, int start, int count);
    void sample_gpu_memory();
};
