cmake_minimum_required(VERSION 3.16)

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
target_compile_options(mon_projet PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=64>)
find_package(assimp REQUIRED)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(mon_projet PRIVATE
    assimp::assimp
)

set_target_properties(mon_projet PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)
