#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct MNISTDataset {
    std::vector<float> images;   // [num_samples * 784], valores en [0, 1]
    std::vector<uint8_t> labels; // [num_samples]
    int num_samples = 0;
    static constexpr int IMAGE_SIZE = 28 * 28;
};

class MNISTLoader {
public:
    static MNISTDataset load_images(const std::string& image_path,
                                    const std::string& label_path);
};
