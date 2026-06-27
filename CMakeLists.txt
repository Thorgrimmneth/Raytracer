cmake_minimum_required(VERSION 3.16)

project(mon_projet LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(OPTIX_ROOT "/home/ejouenne/Downloads/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64")
set(PTX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/ptx")
set(OPTIX_DEVICE_DIR ${CMAKE_SOURCE_DIR}/devicePrograms)

file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})

set(PTX_FILE "${PTX_OUTPUT_DIR}/${FILE_NAME}.ptx")

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

file(GLOB OPTIX_DEVICE_SOURCES
    "${OPTIX_DEVICE_DIR}/*.cu"
)

set(PTX_FILES)

foreach(CUDA_FILE ${OPTIX_DEVICE_SOURCES})

    get_filename_component(FILE_NAME ${CUDA_FILE} NAME_WE)

    set(PTX_FILE "${CMAKE_BINARY_DIR}/${FILE_NAME}.ptx")

    list(APPEND PTX_FILES ${PTX_FILE})

    add_custom_command(
        OUTPUT ${PTX_FILE}

        COMMAND ${CMAKE_CUDA_COMPILER}
            --ptx
            -std=c++17
            -I${OPTIX_ROOT}/include
            -I${OPTIX_DEVICE_DIR}
            ${CUDA_FILE}
            -o ${PTX_FILE}

        DEPENDS ${CUDA_FILE}

        VERBATIM
    )

endforeach()

add_custom_target(
    optix_shaders
    DEPENDS ${PTX_FILES}
)

add_dependencies(mon_projet optix_shaders)