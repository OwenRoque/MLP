#include "training/trainer.hpp"
#include "core/cuda_utils.cuh"
#include "loss/softmax_cross_entropy.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

Trainer::Trainer(Network& network, const TrainConfig& config)
    : network_(network),
      config_(config),
      d_batch_input_(static_cast<std::size_t>(config.batch_size * MNISTDataset::IMAGE_SIZE)),
      d_grad_output_(static_cast<std::size_t>(config.batch_size * network.output_size())),
      max_batch_size_(config.batch_size) {
    CUDA_CHECK(cudaMalloc(&d_batch_labels_, static_cast<std::size_t>(max_batch_size_) * sizeof(int)));
}

Trainer::~Trainer() {
    if (d_batch_labels_ != nullptr) {
        cudaFree(d_batch_labels_);
    }
}

void Trainer::upload_batch(const MNISTDataset& dataset, int start, int count) {
    std::vector<float> host_input(static_cast<std::size_t>(count * MNISTDataset::IMAGE_SIZE));
    std::vector<int> host_labels(static_cast<std::size_t>(count));

    for (int i = 0; i < count; ++i) {
        const int idx = start + i;
        const auto img_offset =
            static_cast<std::size_t>(idx * MNISTDataset::IMAGE_SIZE);
        std::copy(dataset.images.begin() + img_offset,
                  dataset.images.begin() + img_offset + MNISTDataset::IMAGE_SIZE,
                  host_input.begin() + static_cast<std::size_t>(i * MNISTDataset::IMAGE_SIZE));
        host_labels[static_cast<std::size_t>(i)] = static_cast<int>(dataset.labels[static_cast<std::size_t>(idx)]);
    }

    CUDA_CHECK(cudaMemcpy(d_batch_input_.device_data(), host_input.data(),
                          host_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_batch_labels_, host_labels.data(),
                          host_labels.size() * sizeof(int), cudaMemcpyHostToDevice));
}

TrainMetrics Trainer::train_epoch(const MNISTDataset& dataset) {
    TrainMetrics metrics{};
    float loss_accum = 0.0f;
    int correct = 0;
    int total = 0;
    int batch_count = 0;

    for (int start = 0; start < dataset.num_samples; start += config_.batch_size) {
        const int count = std::min(config_.batch_size, dataset.num_samples - start);
        upload_batch(dataset, start, count);

        network_.forward(d_batch_input_.device_data(), count);

        const float batch_loss = SoftmaxCrossEntropy::forward(
            network_.output(), d_batch_labels_,
            d_grad_output_.device_data(), count, network_.output_size());

        network_.backward(d_grad_output_.device_data(), count);
        network_.update(config_.learning_rate);

        loss_accum += batch_loss;
        ++batch_count;

        for (int i = 0; i < count; ++i) {
            const float* logits =
                network_.output() + static_cast<std::ptrdiff_t>(i * network_.output_size());
            const int pred = SoftmaxCrossEntropy::predict_class(logits, network_.output_size());
            const int label = static_cast<int>(dataset.labels[static_cast<std::size_t>(start + i)]);
            correct += (pred == label) ? 1 : 0;
            ++total;
        }

        if (batch_count % config_.log_every == 0) {
            std::cout << "  batch " << batch_count << " — loss: " << batch_loss << '\n';
        }
    }

    metrics.loss = loss_accum / static_cast<float>(batch_count);
    metrics.accuracy = static_cast<float>(correct) / static_cast<float>(total);
    return metrics;
}

TrainMetrics Trainer::evaluate(const MNISTDataset& dataset) {
    TrainMetrics metrics{};
    float loss_accum = 0.0f;
    int correct = 0;
    int batch_count = 0;

    for (int start = 0; start < dataset.num_samples; start += config_.batch_size) {
        const int count = std::min(config_.batch_size, dataset.num_samples - start);
        upload_batch(dataset, start, count);

        network_.forward(d_batch_input_.device_data(), count);

        const float batch_loss = SoftmaxCrossEntropy::forward(
            network_.output(), d_batch_labels_,
            d_grad_output_.device_data(), count, network_.output_size());

        loss_accum += batch_loss;
        ++batch_count;

        for (int i = 0; i < count; ++i) {
            const float* logits =
                network_.output() + static_cast<std::ptrdiff_t>(i * network_.output_size());
            const int pred = SoftmaxCrossEntropy::predict_class(logits, network_.output_size());
            const int label = static_cast<int>(dataset.labels[static_cast<std::size_t>(start + i)]);
            correct += (pred == label) ? 1 : 0;
        }
    }

    metrics.loss = loss_accum / static_cast<float>(batch_count);
    metrics.accuracy = static_cast<float>(correct) / static_cast<float>(dataset.num_samples);
    return metrics;
}
