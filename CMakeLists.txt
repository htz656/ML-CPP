cmake_minimum_required(VERSION 3.30.4)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(ML-CPP VERSION 0.1 LANGUAGES CXX)

# 构建类型和输出目录
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# # 包含目录配置
# include_directories(
#     ${PROJECT_SOURCE_DIR}/include
# )

# 核心库
add_library(ml_core
    src/core/base_model.cpp
    src/core/supervised.cpp
    src/core/unsupervised.cpp
    src/core/regression.cpp
    src/core/classification.cpp
)
target_include_directories(ml_core PUBLIC ${PROJECT_SOURCE_DIR}/include)

# 模型库补充更多源文件
add_library(ml_models
    src/models/linear_regression.cpp
    # 
)
target_link_libraries(ml_models PUBLIC ml_core)

option(BUILD_TESTS "Build test cases" ON)
if(BUILD_TESTS)
    add_executable(test_linear_regression tests/test_linear_regression.cpp)
    target_link_libraries(test_linear_regression PRIVATE ml_models)
endif()