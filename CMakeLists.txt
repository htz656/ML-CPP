cmake_minimum_required(VERSION 3.30.4)

project(ML-cpp VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE "Debug")

include_directories(${PROJECT_SOURCE_DIR}/src/include/)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/ Files)

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/build)