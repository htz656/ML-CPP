// File: include/ml/models/linear_regression.h
#pragma once
#include "ml/core/regression.h"

namespace ml {

class LinearRegression final : public RegressionModel {
public:
    explicit LinearRegression(
        double learning_rate = 0.01, 
        int iterations = 1000);
    
    void train(
        const std::vector<std::vector<double>>& features,
        const std::vector<double>& labels) override;
    
    double predict(const std::vector<double>& features) const override;
    
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    void gradient_descent(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y);
    
    std::vector<double> weights_;
    
    const double learning_rate_;
    const int max_iterations_;
};

} // namespace ml