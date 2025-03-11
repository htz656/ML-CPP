// File: include/ml/core/supervised.h
#pragma once
#include "base_model.h"

namespace ml {
class SupervisedModel : public MachineLearningModel {
public:
    virtual void train(
        const std::vector<std::vector<double>>& features,
        const std::vector<double>& labels) = 0;

    virtual double predict(const std::vector<double>& features) const = 0;
    
    virtual double evaluate(
        const std::vector<std::vector<double>>& test_features,
        const std::vector<double>& test_labels) const;
};
}