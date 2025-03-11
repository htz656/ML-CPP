// File: include/ml/core/regression.h
#pragma once
#include "supervised.h"

namespace ml {

// Base class for all regression models.
class RegressionModel : public SupervisedModel {
public:
    double evaluate(
        const std::vector<std::vector<double>>& test_features,
        const std::vector<double>& test_labels) const override;
};

} // namespace ml