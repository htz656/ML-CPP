// File: include/ml/core/supervised.h
#pragma once
#include "base_model.h"

namespace ml {

// Base class for all supervised learning models.
class SupervisedModel : public MachineLearningModel {
public:
    virtual void train(
        const std::vector<std::vector<double>>& features,
        const std::vector<double>& labels) = 0;

    virtual double predict(const std::vector<double>& features) const = 0;
    
    virtual double evaluate(
        const std::vector<std::vector<double>>& test_features,
        const std::vector<double>& test_labels) const = 0;

    // Validate the input data.
    void check_valid(
        const std::vector<std::vector<double>>& features, 
        const std::vector<double>& labels
    ) const {
        validate(features, labels);
    }

protected:
    void validate(
        const std::vector<std::vector<double>>& features, 
        const std::vector<double>& labels
    ) const {
        validate_features(features);

        if (labels.empty()) {
            throw std::invalid_argument("[ERROR] Label vector is empty.");
        }

        if (features.size() != labels.size()) {
            throw std::invalid_argument("[ERROR] Number of labels does not match the number of features.");
        }
    }
};

} // namespace ml