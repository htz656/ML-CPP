// File: include/ml/core/classification.h
#pragma once
#include "supervised.h"

namespace ml {

// Base class for all classification models.
class ClassificationModel : public SupervisedModel {
public:
    virtual int predict_class(const std::vector<double>& features) const = 0;
    virtual std::vector<double> predict_proba(const std::vector<double>& features) const = 0;

    double predict(const std::vector<double>& features) const override {
        return static_cast<double>(predict_class(features));
    }
    
    double evaluate(
        const std::vector<std::vector<double>>& test_features,
        const std::vector<double>& test_labels) const override;
};

} // namespace ml