cmake_minimum_required(VERSION 3.16)

project(mon_projet LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(OPTIX_ROOT "/home/ejouenne/Downloads/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64")

set(OPTIX_DEVICE_DIR
    ${CMAKE_SOURCE_DIR}/devicePrograms
)

set(RAYGEN_PTX
    ${CMAKE_BINARY_DIR}/raygen.ptx
)

set(MISS_PTX
    ${CMAKE_BINARY_DIR}/miss.ptx
)

set(HIT_PTX
    ${CMAKE_BINARY_DIR}/closesthit.ptx
)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(assimp REQUIRED)

file(GLOB_RECURSE SOURCES
    src/*.cpp
    src/*.cu
    ${OPTIX_DEVICE_DIR}/optix*.cpp
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
    ${OPTIX_ROOT}/include
)

target_link_libraries(mon_projet PRIVATE
    assimp::assimp
    CUDA::cudart
    CUDA::cuda_driver
    glfw
    OpenGL::GL
)

target_compile_options(mon_projet PRIVATE
    #$<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=72>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    #$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

set_target_properties(mon_projet PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 120
)

add_custom_command(
    OUTPUT ${RAYGEN_PTX}

    COMMAND ${CMAKE_CUDA_COMPILER}
        --ptx
        -std=c++17
        -I${OPTIX_ROOT}/include
        -I${OPTIX_DEVICE_DIR}
        ${OPTIX_DEVICE_DIR}/raygen.cu
        -o ${RAYGEN_PTX}

    DEPENDS
        ${OPTIX_DEVICE_DIR}/raygen.cu
)

add_custom_command(
    OUTPUT ${MISS_PTX}

    COMMAND ${CMAKE_CUDA_COMPILER}
        --ptx
        -std=c++17
        -I${OPTIX_ROOT}/include
        -I${OPTIX_DEVICE_DIR}
        ${OPTIX_DEVICE_DIR}/miss.cu
        -o ${MISS_PTX}

    DEPENDS
        ${OPTIX_DEVICE_DIR}/miss.cu
)

add_custom_command(
    OUTPUT ${HIT_PTX}

    COMMAND ${CMAKE_CUDA_COMPILER}
        --ptx
        -std=c++17
        -I${OPTIX_ROOT}/include
        -I${OPTIX_DEVICE_DIR}
        ${OPTIX_DEVICE_DIR}/closesthit.cu
        -o ${HIT_PTX}

    DEPENDS
        ${OPTIX_DEVICE_DIR}/closesthit.cu
)

add_custom_target(
    optix_shaders
    DEPENDS
        ${RAYGEN_PTX}
        ${MISS_PTX}
        ${HIT_PTX}
)

add_dependencies(
    mon_projet
    optix_shaders
)