#pragma once

#include "activations/activation.hpp"
#include "layers/activation_layer.hpp"
#include "layers/linear_layer.hpp"
#include "network/network.hpp"

#include <memory>
#include <ostream>
#include <stdexcept>

struct NetworkConfig {
    int input_size = 784;
    int output_size = 10;
    int hidden_size = 128; // 0 = perceptron lineal (sin capa oculta)
    ActivationType activation = ActivationType::ReLU;
    unsigned int seed = 42;
};

inline Network build_network(const NetworkConfig& config) {
    Network network;

    if (config.hidden_size <= 0) {
        network.add_layer(
            std::make_unique<LinearLayer>(config.input_size, config.output_size, config.seed));
        return network;
    }

    network.add_layer(std::make_unique<LinearLayer>(config.input_size, config.hidden_size,
                                                    config.seed));
    network.add_layer(
        std::make_unique<ActivationLayer>(config.hidden_size, config.activation));
    network.add_layer(std::make_unique<LinearLayer>(config.hidden_size, config.output_size,
                                                    config.seed + 1));
    return network;
}

inline void print_network_architecture(const Network& network, std::ostream& out,
                                       ActivationType activation, int hidden_size) {
    if (hidden_size <= 0) {
        out << "Arquitectura: Linear(" << network.input_size() << " -> "
            << network.output_size() << ") [perceptron lineal]\n";
        return;
    }

    out << "Arquitectura: Linear(" << network.input_size() << " -> " << hidden_size
        << ") -> " << activation_name(activation) << " -> Linear(" << hidden_size << " -> "
        << network.output_size() << ")\n";
}
