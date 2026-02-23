#pragma once

#include "../cuda_scene.cuh"
#include "../raytracingUtils/cuda_ray.cuh"
#include "../raytracingUtils/cuda_hitrecord.cuh"
#include "cuda_direct_lighting_integrator.cuh"
#include "../utils/cuda_defines.cuh"

struct CurrentLight
{
    Ray ray;
    float3 weight;
    bool inside;
    int depth;
};

struct WhittedIntegrator
{
    __device__ static float3 lighting(
        const CudaScene &scene,
        const Ray &primaryRay,
        const float tMin,
        const float tMax,
        curandState *rng);

    __device__ static float3 toneMap(const float3 &c);

    __device__ static float3 getSkyColor(const Ray &p_ray);
};