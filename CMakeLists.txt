cmake_minimum_required(VERSION 3.16)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.2/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-12)
set(CMAKE_CUDA_ARCHITECTURES 120)
project(mon_projet LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

file(GLOB_RECURSE SOURCES
    src/*.cpp
    src/*.cu
)

add_executable(mon_projet ${SOURCES})

target_include_directories(mon_projet PRIVATE
    src
    lib
)
target_compile_options(mon_projet PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --maxrregcount=64
        -Xcudafe=--diag_suppress=20012
    >
)
find_package(assimp REQUIRED)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(mon_projet PRIVATE
    assimp::assimp
)

set_target_properties(mon_projet PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
