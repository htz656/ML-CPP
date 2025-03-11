// File: src/core/regression.cpp
#include "ml/core/regression.h"

namespace ml {

double RegressionModel::evaluate(
    const std::vector<std::vector<double>>& test_features,
    const std::vector<double>& test_labels
) const {
    check_valid(test_features, test_labels);

    double mse = 0.0;
    for (size_t i = 0; i < test_features.size(); ++i) {
        const double error = predict(test_features[i]) - test_labels[i];
        mse += error * error;
    }
    return mse / static_cast<double>(test_features.size());
}

} // namespace ml