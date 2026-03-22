cmake_minimum_required(VERSION 3.10)
project(BitMamba2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags — platform-adaptive
if(APPLE)
    # AppleClang: use -Xclang -fopenmp with Homebrew libomp
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcpu=native -Xclang -fopenmp")
    include_directories(/opt/homebrew/opt/libomp/include)
    link_directories(/opt/homebrew/opt/libomp/lib)
    set(OPENMP_LIB omp)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")
    set(OPENMP_LIB "")
endif()

# Include directories
include_directories(include)
include_directories(src) # For tokenizer.h

# Core library
add_library(bitmamba_lib
    src/basic_utils.cpp
    src/quantization.cpp
    src/kernels.cpp
    src/block.cpp
    src/model.cpp
)

# CLI Executable
add_executable(bitmamba
    examples/main.cpp
)

target_link_libraries(bitmamba bitmamba_lib ${OPENMP_LIB})
