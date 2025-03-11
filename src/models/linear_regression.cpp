#include "ml/models/linear_regression.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ml {

LinearRegression::LinearRegression(double learning_rate, int iterations) : 
    learning_rate_(learning_rate), 
    max_iterations_(iterations) {}

void LinearRegression::train(
    const std::vector<std::vector<double>>& features,
    const std::vector<double>& labels
) {
    check_valid(features, labels);
    
    std::vector<std::vector<double>> X = features;
    for (auto& row : X) row.insert(row.begin(), 1.0); 
    
    weights_.resize(X[0].size(), 0.0);
    
    gradient_descent(X, labels);
}

double LinearRegression::predict(const std::vector<double>& features) const {
    if (features.size() + 1 != weights_.size()) { 
        throw std::invalid_argument("[ERROR] Feature dimension does not match the model");
    }
    
    double prediction = weights_[0];    // Start with the bias term
    for (size_t i = 0; i < features.size(); ++i) {
        prediction += weights_[i + 1] * features[i]; // Skip the bias weight
    }
    return prediction;
}

void LinearRegression::gradient_descent(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y
) {
    const size_t m = X.size();
    const size_t n = X[0].size();
    
    for (int iter = 0; iter < max_iterations_; ++iter) {
        std::vector<double> grad(n, 0.0);
        
        // Compute gradient
        for (size_t i = 0; i < m; ++i) {
            double prediction = 0.0;
            for (size_t j = 0; j < n; ++j) {
                prediction += X[i][j] * weights_[j];
            }
            const double error = prediction - y[i];
            for (size_t j = 0; j < n; ++j) {
                grad[j] += error * X[i][j];
            }
        }
        
        // Update weights
        for (size_t j = 0; j < n; ++j) {
            weights_[j] -= (learning_rate_ / m) * grad[j];
        }
    }
}

void LinearRegression::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("[ERROR] Unable to open file");
    
    file << "LinearRegression\n";
    file << weights_.size() << "\n";
    for (const auto& w : weights_) file << w << " ";
    file << "\n";
}

void LinearRegression::load(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("[ERROR] Unable to open file");
    
    std::string identifier;
    std::getline(file, identifier);
    if (identifier != "LinearRegression") {
        throw std::runtime_error("[ERROR] File format is incompatible");
    }
    
    size_t size;
    file >> size;
    weights_.resize(size);
    for (size_t i = 0; i < size; ++i) file >> weights_[i];
}

} // namespace ml