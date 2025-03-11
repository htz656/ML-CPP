// File: src/core/base_model.cpp
#include "ml/core/base_model.h"
#include <stdexcept>

namespace ml {
void MachineLearningModel::validate_features(
    const std::vector<std::vector<double>>& features) const {
    if (features.empty()) throw std::invalid_argument("Empty features");
    const size_t dim = features[0].size();
    for (const auto& vec : features) {
        if (vec.size() != dim) throw std::invalid_argument("Feature dimension mismatch");
    }
}
}