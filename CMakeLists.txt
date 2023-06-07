cmake_minimum_required(VERSION 3.24)

project(matrix_mul CXX)

add_executable(bench bench.cpp)

target_compile_features(bench PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(bench PUBLIC /W4 /WX)
else()
    target_compile_options(bench PUBLIC -Wall -Wextra -Werror -pedantic -mavx -O3 -march=native -ffast-math)
endif()

find_package(Benchmark REQUIRED)

target_link_libraries(bench
    benchmark::benchmark
    benchmark::benchmark_main
)
