// File: include/ml/core/unsupervised.h
#pragma once
#include "base_model.h"

namespace ml {

// Base class for all unsupervised models.
class UnsupervisedModel : public MachineLearningModel {
public:
    virtual void train(const std::vector<std::vector<double>>& features) = 0;

    // Validate the input data.
    void check_valid(const std::vector<std::vector<double>>& features) const {
        validate_features(features);
    }
};

// Base class for all dimensionality reduction models.
class DimensionalityReductionModel : public UnsupervisedModel {
public:
    virtual std::vector<std::vector<double>> transform(const std::vector<std::vector<double>>& features) const = 0;
};


// Base class for all clustering models.
class ClusteringModel : public UnsupervisedModel {
public:
    virtual std::vector<int> predict_cluster(const std::vector<std::vector<double>>& features) const = 0;
    
    virtual double silhouette_score(const std::vector<std::vector<double>>& features) const = 0;
};

} // namespace ml