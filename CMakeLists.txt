cmake_minimum_required(VERSION 3.30.4)
project(ML-CPP VERSION 0.1 LANGUAGES CXX)

# 基础配置
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 构建类型和输出目录
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# 包含目录配置
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# 核心库(基础模型、监督/非监督学习基类)
add_library(ml_core
    src/core/base_model.cpp
    src/core/supervised.cpp
    src/core/unsupervised.cpp
    src/core/regression.cpp
    src/core/classification.cpp
)

# 模型库(具体算法实现)
add_library(ml_models
    src/models/linear_regression.cpp
    src/models/logistic_regression.cpp
    # 添加其他模型源文件...
)
target_link_libraries(ml_models PUBLIC ml_core)

# # 单元测试配置
# option(BUILD_TESTS "Build test cases" ON)
# if(BUILD_TESTS)
#     enable_testing()
    
#     # 查找测试框架（示例使用Catch2）
#     find_package(Catch2 REQUIRED)
    
#     # 添加测试可执行文件
#     add_executable(ml_tests
#         tests/test_linear_regression.cpp
#         tests/test_logistic_regression.cpp
#         # 添加其他测试文件...
#     )
#     target_link_libraries(ml_tests
#         PRIVATE 
#         ml_models
#         Catch2::Catch2WithMain
#     )
    
#     # 注册测试用例
#     add_test(NAME ml_tests COMMAND ml_tests)
# endif()

# # 安装规则（可选）
# install(DIRECTORY include/ DESTINATION include)
# install(TARGETS ml_core ml_models
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
#     RUNTIME DESTINATION bin
# )