#include "data/mnist_loader.hpp"
#include "layers/linear_layer.hpp"
#include "network/network.hpp"
#include "training/trainer.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

struct CliOptions {
    std::string data_dir = "data/mnist";
    int epochs = 10;
    int batch_size = 128;
    float learning_rate = 0.1f;
};

bool has_mnist_files(const std::string& dir) {
    const std::filesystem::path base(dir);
    return std::filesystem::exists(base / "train-images-idx3-ubyte") &&
           std::filesystem::exists(base / "train-labels-idx1-ubyte") &&
           std::filesystem::exists(base / "t10k-images-idx3-ubyte") &&
           std::filesystem::exists(base / "t10k-labels-idx1-ubyte");
}

std::string resolve_data_dir(const std::string& requested) {
    if (has_mnist_files(requested)) {
        return requested;
    }

    const std::string from_build = "../" + requested;
    if (has_mnist_files(from_build)) {
        return from_build;
    }

    return requested;
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            opts.data_dir = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            opts.epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            opts.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            opts.learning_rate = std::stof(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Uso: mnist_perceptron [--data DIR] [--epochs N] "
                         "[--batch-size N] [--lr RATE]\n";
            std::exit(0);
        }
    }
    return opts;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions opts = parse_args(argc, argv);
        const std::string data_dir = resolve_data_dir(opts.data_dir);

        std::cout << "=== Perceptrón MNIST (CUDA) ===\n";
        std::cout << "Cargando datos desde: " << data_dir << '\n';

        if (!has_mnist_files(data_dir)) {
            throw std::runtime_error(
                "No se encontraron archivos MNIST en '" + data_dir +
                "'. Ejecute scripts/download_mnist.sh o indique --data con la ruta correcta.");
        }

        const auto train_images = data_dir + "/train-images-idx3-ubyte";
        const auto train_labels = data_dir + "/train-labels-idx1-ubyte";
        const auto test_images = data_dir + "/t10k-images-idx3-ubyte";
        const auto test_labels = data_dir + "/t10k-labels-idx1-ubyte";

        const MNISTDataset train_set = MNISTLoader::load_images(train_images, train_labels);
        const MNISTDataset test_set = MNISTLoader::load_images(test_images, test_labels);

        std::cout << "Train: " << train_set.num_samples << " muestras\n";
        std::cout << "Test:  " << test_set.num_samples << " muestras\n";

        Network network;
        network.add_layer(std::make_unique<LinearLayer>(MNISTDataset::IMAGE_SIZE, 10));

        TrainConfig config;
        config.epochs = opts.epochs;
        config.batch_size = opts.batch_size;
        config.learning_rate = opts.learning_rate;

        Trainer trainer(network, config);

        for (int epoch = 1; epoch <= config.epochs; ++epoch) {
            std::cout << "\nÉpoca " << epoch << '/' << config.epochs << '\n';
            const auto train_metrics = trainer.train_epoch(train_set);
            const auto test_metrics = trainer.evaluate(test_set);

            std::cout << "  train loss: " << train_metrics.loss
                      << " | train acc: " << (train_metrics.accuracy * 100.0f) << "%\n";
            std::cout << "  test  loss: " << test_metrics.loss
                      << " | test  acc: " << (test_metrics.accuracy * 100.0f) << "%\n";
        }

        std::cout << "\nEntrenamiento completado.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
