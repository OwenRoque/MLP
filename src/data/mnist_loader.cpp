#include "data/mnist_loader.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

uint32_t read_be_u32(std::ifstream& file) {
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    if (!file) {
        throw std::runtime_error("Error leyendo archivo MNIST");
    }
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) |
           static_cast<uint32_t>(bytes[3]);
}

void skip_mnist_header(std::ifstream& file, bool is_image) {
    read_be_u32(file); // magic
    read_be_u32(file); // num items
    if (is_image) {
        read_be_u32(file); // rows
        read_be_u32(file); // cols
    }
}

} // namespace

MNISTDataset MNISTLoader::load_images(const std::string& image_path,
                                      const std::string& label_path) {
    std::ifstream image_file(image_path, std::ios::binary);
    std::ifstream label_file(label_path, std::ios::binary);

    if (!image_file || !label_file) {
        throw std::runtime_error(
            "No se pudieron abrir los archivos MNIST en:\n  " + image_path + "\n  " +
            label_path + "\nVerifique la ruta con --data o ejecute scripts/download_mnist.sh");
    }

    skip_mnist_header(image_file, true);
    skip_mnist_header(label_file, false);

    image_file.seekg(0, std::ios::end);
    const auto image_bytes = image_file.tellg();
    image_file.seekg(16, std::ios::beg);

    label_file.seekg(0, std::ios::end);
    const auto label_bytes = label_file.tellg();
    label_file.seekg(8, std::ios::beg);

    const int num_samples =
        static_cast<int>(static_cast<long long>(label_bytes) - 8);
    const int expected_image_bytes = num_samples * MNISTDataset::IMAGE_SIZE;

    if (static_cast<long long>(image_bytes) - 16 != expected_image_bytes) {
        throw std::runtime_error("Tamaño de imágenes MNIST inconsistente con etiquetas");
    }

    MNISTDataset dataset;
    dataset.num_samples = num_samples;
    dataset.labels.resize(static_cast<std::size_t>(num_samples));
    dataset.images.resize(static_cast<std::size_t>(num_samples * MNISTDataset::IMAGE_SIZE));

    std::vector<uint8_t> raw_images(static_cast<std::size_t>(expected_image_bytes));
    image_file.read(reinterpret_cast<char*>(raw_images.data()), expected_image_bytes);
    label_file.read(reinterpret_cast<char*>(dataset.labels.data()), num_samples);

    for (int i = 0; i < expected_image_bytes; ++i) {
        dataset.images[static_cast<std::size_t>(i)] =
            static_cast<float>(raw_images[static_cast<std::size_t>(i)]) / 255.0f;
    }

    return dataset;
}
