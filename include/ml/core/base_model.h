// File: include/ml/core/base_model.h
#pragma once
#include <vector>
#include <string>

namespace ml {
class MachineLearningModel {
public:
    virtual ~MachineLearningModel() = default;
    
    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;

protected:
    void validate_features(const std::vector<std::vector<double>>& features) const;
};
}