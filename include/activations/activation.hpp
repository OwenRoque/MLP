#pragma once

#include <stdexcept>
#include <string>

enum class ActivationType { ReLU, GeLU, Sigmoid };

inline ActivationType parse_activation(const std::string& name) {
    if (name == "relu" || name == "ReLU") {
        return ActivationType::ReLU;
    }
    if (name == "gelu" || name == "GeLU") {
        return ActivationType::GeLU;
    }
    if (name == "sigmoid" || name == "Sigmoid") {
        return ActivationType::Sigmoid;
    }
    throw std::invalid_argument("Activacion desconocida: " + name +
                                " (opciones: relu, gelu, sigmoid)");
}

inline const char* activation_name(ActivationType type) {
    switch (type) {
    case ActivationType::ReLU:
        return "ReLU";
    case ActivationType::GeLU:
        return "GeLU";
    case ActivationType::Sigmoid:
        return "Sigmoid";
    }
    return "Unknown";
}
