#include "inference/inference_export.hpp"
#include "core/cuda_utils.cuh"
#include "core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

namespace {

void append_u16(std::vector<char>& buf, uint16_t v) {
    buf.push_back(static_cast<char>(v & 0xff));
    buf.push_back(static_cast<char>((v >> 8) & 0xff));
}

void append_u32(std::vector<char>& buf, uint32_t v) {
    buf.push_back(static_cast<char>(v & 0xff));
    buf.push_back(static_cast<char>((v >> 8) & 0xff));
    buf.push_back(static_cast<char>((v >> 16) & 0xff));
    buf.push_back(static_cast<char>((v >> 24) & 0xff));
}

void append_str(std::vector<char>& buf, const std::string& s) {
    buf.insert(buf.end(), s.begin(), s.end());
}

std::vector<char> make_npy_header(const std::string& descr, const std::vector<size_t>& shape) {
    std::ostringstream dict;
    dict << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            dict << ", ";
        }
        dict << shape[i];
    }
    if (shape.size() == 1) {
        dict << ",";
    }
    dict << "), }";

    std::string dict_str = dict.str();
    const size_t preamble = 10;
    const size_t padding = (16 - (preamble + dict_str.size()) % 16) % 16;
    dict_str.append(padding, ' ');
    dict_str.back() = '\n';

    std::vector<char> header;
    header.push_back(static_cast<char>(0x93));
    append_str(header, "NUMPY");
    header.push_back(static_cast<char>(0x01));
    header.push_back(static_cast<char>(0x00));
    append_u16(header, static_cast<uint16_t>(dict_str.size()));
    append_str(header, dict_str);
    return header;
}

struct NpyEntry {
    std::string name;
    std::vector<char> header;
    const char* data = nullptr;
    size_t data_bytes = 0;
};

void write_npz(const std::string& path, const std::vector<NpyEntry>& entries) {
    if (entries.empty()) {
        throw std::runtime_error("write_npz: sin arrays");
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("No se pudo escribir: " + path);
    }

    std::vector<std::vector<char>> local_headers;
    std::vector<std::vector<char>> global_headers;
    local_headers.reserve(entries.size());
    global_headers.reserve(entries.size());

    size_t offset = 0;

    for (const auto& entry : entries) {
        const std::string fname = entry.name + ".npy";
        const size_t nbytes = entry.header.size() + entry.data_bytes;

        uint32_t crc = crc32(0L, reinterpret_cast<const Bytef*>(entry.header.data()),
                             static_cast<uInt>(entry.header.size()));
        crc = crc32(crc, reinterpret_cast<const Bytef*>(entry.data),
                    static_cast<uInt>(entry.data_bytes));

        std::vector<char> local;
        append_str(local, "PK");
        append_u16(local, 0x0403);
        append_u16(local, 20);
        append_u16(local, 0);
        append_u16(local, 0);
        append_u16(local, 0);
        append_u16(local, 0);
        append_u32(local, crc);
        append_u32(local, static_cast<uint32_t>(nbytes));
        append_u32(local, static_cast<uint32_t>(nbytes));
        append_u16(local, static_cast<uint16_t>(fname.size()));
        append_u16(local, 0);
        append_str(local, fname);

        std::vector<char> global;
        append_str(global, "PK");
        append_u16(global, 0x0201);
        append_u16(global, 20);
        global.insert(global.end(), local.begin() + 4, local.begin() + 30);
        append_u16(global, 0);
        append_u16(global, 0);
        append_u16(global, 0);
        append_u32(global, 0);
        append_u32(global, static_cast<uint32_t>(offset));
        append_str(global, fname);

        out.write(local.data(), static_cast<std::streamsize>(local.size()));
        out.write(entry.header.data(), static_cast<std::streamsize>(entry.header.size()));
        out.write(entry.data, static_cast<std::streamsize>(entry.data_bytes));

        local_headers.push_back(std::move(local));
        global_headers.push_back(std::move(global));

        offset += local_headers.back().size() + nbytes;
    }

    const size_t central_dir_offset = offset;
    size_t central_dir_size = 0;
    for (const auto& global : global_headers) {
        out.write(global.data(), static_cast<std::streamsize>(global.size()));
        central_dir_size += global.size();
    }

    std::vector<char> footer;
    append_str(footer, "PK");
    append_u16(footer, 0x0605);
    append_u16(footer, 0);
    append_u16(footer, 0);
    append_u16(footer, static_cast<uint16_t>(entries.size()));
    append_u16(footer, static_cast<uint16_t>(entries.size()));
    append_u32(footer, static_cast<uint32_t>(central_dir_size));
    append_u32(footer, static_cast<uint32_t>(central_dir_offset));
    append_u16(footer, 0);

    out.write(footer.data(), static_cast<std::streamsize>(footer.size()));
}

void softmax_cpu(const float* logits, float* probs, int num_classes) {
    float max_logit = logits[0];
    for (int c = 1; c < num_classes; ++c) {
        max_logit = std::max(max_logit, logits[c]);
    }

    float sum = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
        probs[c] = std::exp(logits[c] - max_logit);
        sum += probs[c];
    }

    for (int c = 0; c < num_classes; ++c) {
        probs[c] /= sum;
    }
}

} // namespace

void export_inference_results(Network& network, const MNISTDataset& test_set,
                              const std::string& output_path, int num_samples,
                              int start_index) {
    if (test_set.num_samples == 0) {
        throw std::runtime_error("Conjunto de test vacio");
    }

    const int count = std::min(num_samples, test_set.num_samples - start_index);
    if (count <= 0) {
        throw std::runtime_error("Rango de muestras invalido para exportar inferencia");
    }

    Tensor d_input(static_cast<std::size_t>(count * MNISTDataset::IMAGE_SIZE));
    std::vector<float> host_input(static_cast<std::size_t>(count * MNISTDataset::IMAGE_SIZE));
    std::vector<float> images(static_cast<std::size_t>(count * MNISTDataset::IMAGE_SIZE));
    std::vector<int32_t> y_true(static_cast<std::size_t>(count));
    std::vector<int32_t> y_pred(static_cast<std::size_t>(count));
    std::vector<float> confidence(static_cast<std::size_t>(count));

    for (int i = 0; i < count; ++i) {
        const int idx = start_index + i;
        const auto offset = static_cast<std::size_t>(idx * MNISTDataset::IMAGE_SIZE);
        std::copy(test_set.images.begin() + offset,
                  test_set.images.begin() + offset + MNISTDataset::IMAGE_SIZE,
                  host_input.begin() + static_cast<std::size_t>(i * MNISTDataset::IMAGE_SIZE));
        std::copy(test_set.images.begin() + offset,
                  test_set.images.begin() + offset + MNISTDataset::IMAGE_SIZE,
                  images.begin() + static_cast<std::size_t>(i * MNISTDataset::IMAGE_SIZE));
        y_true[static_cast<std::size_t>(i)] =
            static_cast<int32_t>(test_set.labels[static_cast<std::size_t>(idx)]);
    }

    CUDA_CHECK(cudaMemcpy(d_input.device_data(), host_input.data(),
                          host_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    network.forward(d_input.device_data(), count);

    const int num_classes = network.output_size();
    std::vector<float> host_logits(static_cast<std::size_t>(num_classes));
    std::vector<float> probs(static_cast<std::size_t>(num_classes));

    for (int i = 0; i < count; ++i) {
        const float* logits =
            network.output() + static_cast<std::ptrdiff_t>(i * num_classes);
        CUDA_CHECK(cudaMemcpy(host_logits.data(), logits, num_classes * sizeof(float),
                              cudaMemcpyDeviceToHost));

        softmax_cpu(host_logits.data(), probs.data(), num_classes);

        const int pred = static_cast<int>(std::distance(
            probs.begin(), std::max_element(probs.begin(), probs.end())));
        y_pred[static_cast<std::size_t>(i)] = static_cast<int32_t>(pred);
        confidence[static_cast<std::size_t>(i)] = probs[static_cast<std::size_t>(pred)];
    }

    const std::vector<size_t> images_shape = {static_cast<size_t>(count), 28, 28};
    const std::vector<size_t> vector_shape = {static_cast<size_t>(count)};

    const auto images_header = make_npy_header("<f4", images_shape);
    const auto y_true_header = make_npy_header("<i4", vector_shape);
    const auto y_pred_header = make_npy_header("<i4", vector_shape);
    const auto confidence_header = make_npy_header("<f4", vector_shape);

    const std::vector<NpyEntry> entries = {
        {"images", images_header, reinterpret_cast<const char*>(images.data()),
         images.size() * sizeof(float)},
        {"y_true", y_true_header, reinterpret_cast<const char*>(y_true.data()),
         y_true.size() * sizeof(int32_t)},
        {"y_pred", y_pred_header, reinterpret_cast<const char*>(y_pred.data()),
         y_pred.size() * sizeof(int32_t)},
        {"confidence", confidence_header, reinterpret_cast<const char*>(confidence.data()),
         confidence.size() * sizeof(float)},
    };

    write_npz(output_path, entries);
}
