#pragma once

#include "objectGPU/cuda_lightsample.cuh"
#include "cuda_defines.cuh"
#include "objectGPU/cuda_op.cuh"

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

enum LightType
{
    POINT,
    CYLINDER,
    DIRECTIONNAL,
    QUAD
};

struct Light
{
    float3 position;
    float height;
    float3 color;
    float radius;
    float3 direction;
    float area;
    float3 u;
    float power;
    float3 v;
    LightType type;
    float3 normal;

    __device__ LightSample sampleCylinder(const float3&, curandState*) const;
    __device__ LightSample sampleDirectionnal(const float3&) const;
    __device__ LightSample samplePoint(const float3&) const;
    __device__ LightSample sampleQuad(const float3&, curandState*) const;
    __device__ LightSample sample(const float3&, curandState*) const;
    __device__ LightSample sample(const float3&) const;
};