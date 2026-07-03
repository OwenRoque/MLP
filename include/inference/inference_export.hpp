#pragma once

#include "data/mnist_loader.hpp"
#include "network/network.hpp"

#include <string>

/// Ejecuta inferencia sobre muestras del test y guarda resultados en .npz para Python.
/// Arrays: images [N,28,28], y_true [N], y_pred [N], confidence [N] (float 0–1).
void export_inference_results(Network& network, const MNISTDataset& test_set,
                              const std::string& output_path, int num_samples = 10,
                              int start_index = 0);
