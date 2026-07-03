#pragma once

/// Softmax + entropia cruzada combinados (estable numericamente).
/// En entrenamiento: calcula perdida y deja en output el gradiente ∂L/∂logits.
class SoftmaxCrossEntropy {
public:
    /// logits: [batch * num_classes] en GPU
    /// labels: [batch] enteros en GPU
    /// Devuelve la perdida media del batch.
    static float forward(const float* logits, const int* labels, float* grad_logits,
                         int batch_size, int num_classes);

    /// Para inferencia: devuelve la clase predicha (argmax de logits).
    static int predict_class(const float* logits, int num_classes);
};
