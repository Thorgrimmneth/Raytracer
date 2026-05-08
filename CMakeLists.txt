cmake_minimum_required(VERSION 3.16)

project(mon_projet LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(assimp REQUIRED)

file(GLOB_RECURSE SOURCES
    src/*.cpp
    src/*.cu
)

add_executable(mon_projet
    ${SOURCES}
    lib/glad/src/glad.c
)

set_source_files_properties(
    lib/glad/src/glad.c
    PROPERTIES
    LANGUAGE C
)

target_include_directories(mon_projet PRIVATE
    src
    lib
    lib/glad/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(mon_projet PRIVATE
    assimp::assimp
    CUDA::cudart
    glfw
    OpenGL::GL
)

set_target_properties(mon_projet PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 120
)