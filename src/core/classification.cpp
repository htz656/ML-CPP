// File: src/core/classification.cpp
#include "ml/core/classification.h"
#include <cmath>

namespace ml {

double ClassificationModel::evaluate(
    const std::vector<std::vector<double>>& test_features,
    const std::vector<double>& test_labels
) const {
    check_valid(test_features, test_labels);

    size_t correct = 0;
    for (size_t i = 0; i < test_features.size(); ++i) {
        const int predicted = predict_class(test_features[i]);
        const int actual = static_cast<int>(std::round(test_labels[i]));
        if (predicted == actual) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / test_features.size();
}

} // namespace ml