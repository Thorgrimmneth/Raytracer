#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <optix.h>

#include <sstream>
#include <stdexcept>
#include <iostream>

#ifndef HD_INLINE
#define HD_INLINE __host__ __device__ inline
#endif

#ifndef D_FORCEINLINE
#define D_FORCEINLINE __device__ __forceinline__
#endif

#ifndef H_INLINE
#define H_INLINE __host__ inline
#endif

#ifndef HD_FORCEINLINE
#define HD_FORCEINLINE __host__ __device__ __forceinline__
#endif

#ifndef HD
#define HD __host__ __device__
#endif

#ifndef HOST
#define HOST __host__
#endif

#ifndef DEVICE
#define DEVICE __device__
#endif

#ifndef GLOBAL
#define GLOBAL __global__
#endif

inline void cudaCheck(cudaError_t result, const char *expr, const char *file, int line)
{
    if (result != cudaSuccess)
    {
        std::stringstream ss;

        ss << "CUDA Runtime Error\n"
           << "Expression : " << expr << "\n"
           << "File       : " << file << "\n"
           << "Line       : " << line << "\n"
           << "Message    : " << cudaGetErrorString(result);

        throw std::runtime_error(ss.str());
    }
}

#define CUDA_CHECK(x) cudaCheck((x), #x, __FILE__, __LINE__)

inline void optixCheck(OptixResult result, const char *expr, const char *file, int line)
{
    if (result != OPTIX_SUCCESS)
    {
        std::stringstream ss;

        ss << "OptiX Error\n"
           << "Expression : " << expr << "\n"
           << "File       : " << file << "\n"
           << "Line       : " << line << "\n"
           << "Code       : " << static_cast<int>(result);

        throw std::runtime_error(ss.str());
    }
}

#define OPTIX_CHECK(x) optixCheck((x), #x, __FILE__, __LINE__)

inline void cuCheck(CUresult result, const char *expr, const char *file, int line)
{
    if (result != CUDA_SUCCESS)
    {
        const char *errorName = nullptr;
        const char *errorString = nullptr;

        cuGetErrorName(result, &errorName);

        cuGetErrorString(result, &errorString);

        std::stringstream ss;

        ss << "CUDA Driver Error\n"
           << "Expression : " << expr << "\n"
           << "File       : " << file << "\n"
           << "Line       : " << line << "\n"
           << "Error Name : " << (errorName ? errorName : "Unknown") << "\n"
           << "Message    : " << (errorString ? errorString : "Unknown");

        throw std::runtime_error(ss.str());
    }
}

#define CU_CHECK(x) cuCheck((x), #x, __FILE__, __LINE__)