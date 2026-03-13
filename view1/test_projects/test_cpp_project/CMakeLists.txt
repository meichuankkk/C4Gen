cmake_minimum_required(VERSION 3.16)
project(test_cpp_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_cpp_project
    src/main.cpp
    src/base.cpp
    src/derived.cpp
    src/util.cpp
)

target_include_directories(test_cpp_project PRIVATE include)
