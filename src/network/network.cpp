#include "network/network.hpp"

#include <stdexcept>

void Network::add_layer(std::unique_ptr<Layer> layer) {
    if (!layers_.empty() && layer->input_size() != layers_.back()->output_size()) {
        throw std::runtime_error("Dimensiones incompatibles entre capas consecutivas");
    }
    layers_.push_back(std::move(layer));
}

void Network::forward(const float* input, int batch_size) {
    if (layers_.empty()) {
        throw std::runtime_error("Network sin capas");
    }

    layers_.front()->forward(input, batch_size);
    for (std::size_t i = 1; i < layers_.size(); ++i) {
        layers_[i]->forward(layers_[i - 1]->output(), batch_size);
    }
}

void Network::backward(const float* grad_output, int batch_size) {
    if (layers_.empty()) {
        throw std::runtime_error("Network sin capas");
    }

    if (layers_.size() == 1) {
        layers_.front()->backward(grad_output, batch_size);
        return;
    }

    layers_.back()->backward(grad_output, batch_size);
    for (int i = static_cast<int>(layers_.size()) - 2; i >= 0; --i) {
        layers_[static_cast<std::size_t>(i)]->backward(
            layers_[static_cast<std::size_t>(i + 1)]->grad_input(), batch_size);
    }
}

void Network::update(float learning_rate) {
    for (auto& layer : layers_) {
        layer->update(learning_rate);
    }
}

const float* Network::output() const {
    return layers_.back()->output();
}

const Layer& Network::output_layer() const {
    return *layers_.back();
}

int Network::input_size() const {
    return layers_.empty() ? 0 : layers_.front()->input_size();
}

int Network::output_size() const {
    return layers_.empty() ? 0 : layers_.back()->output_size();
}
