// File: include/ml/core/base_model.h
#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

namespace ml {

// Base class for all machine learning models.
class MachineLearningModel {
public:
    virtual ~MachineLearningModel() = default;
    
    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;

protected:
    void validate_features(const std::vector<std::vector<double>>& features) const {
        if (features.empty()) {
            throw std::invalid_argument("[ERROR] Feature matrix is empty, at least one sample is required.");
        }
        
        const size_t num_features = features[0].size();
        // Check for consistent feature dimensions
        for (size_t i = 0; i < features.size(); ++i) {
            if (features[i].size() != num_features) {
                throw std::invalid_argument(
                    "[ERROR] Inconsistent feature dimensions at sample index:" 
                    + std::to_string(i) 
                    + ", expected number of features:" 
                    + std::to_string(num_features)
                );
            }
            // Check for invalid values (NaN or infinity)
            for (double val : features[i]) {
                if (std::isnan(val) || std::isinf(val)) {
                    throw std::invalid_argument(
                        "[ERROR] Sample " 
                        + std::to_string(i) 
                        + " contains an invalid value: " 
                        + std::to_string(val)
                    );
                }
            }
        }
    }
};

} // namespace ml