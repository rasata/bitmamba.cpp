cmake_minimum_required(VERSION 3.10)
project(BitMamba2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Options ---
option(USE_AVX512 "Enable native AVX-512 kernels (slower on Cascade Lake due to throttling)" OFF)

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

if(USE_AVX512)
    add_compile_definitions(BITMAMBA_USE_AVX512)
    message(STATUS "AVX-512 native kernels ENABLED (opt-in)")
else()
    message(STATUS "AVX-512 native kernels disabled (default: AVX2 path used even on AVX-512 hardware)")
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
