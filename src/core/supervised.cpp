// File: src/core/supervised.cpp
#include "ml/core/supervised.h"
#include <stdexcept>
#include <numeric> // std::accumulate

namespace ml {
double SupervisedModel::evaluate(
    const std::vector<std::vector<double>>& test_features,
    const std::vector<double>& test_labels) const {
    validate_features(test_features);
    if (test_features.size() != test_labels.size()) {
        throw std::invalid_argument("Feature-label size mismatch");
    }

    double total = 0.0;
    for (size_t i = 0; i < test_features.size(); ++i) {
        double error = predict(test_features[i]) - test_labels[i];
        total += error * error; // 默认MSE
    }
    return total / test_features.size();
}
}