#pragma once

/// Softmax + entropía cruzada combinados (estable numéricamente).
/// En entrenamiento: calcula pérdida y deja en output el gradiente ∂L/∂logits.
class SoftmaxCrossEntropy {
public:
    /// logits: [batch * num_classes] en GPU
    /// labels: [batch] enteros en GPU
    /// Devuelve la pérdida media del batch.
    static float forward(const float* logits, const int* labels, float* grad_logits,
                         int batch_size, int num_classes);

    /// Para inferencia: devuelve la clase predicha (argmax de logits).
    static int predict_class(const float* logits, int num_classes);
};
