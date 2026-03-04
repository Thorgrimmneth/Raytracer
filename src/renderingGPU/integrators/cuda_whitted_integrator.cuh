#pragma once

#include "../cuda_scene.cuh"
#include "../raytracingUtils/cuda_ray.cuh"
#include "../raytracingUtils/cuda_hitrecord.cuh"
#include "cuda_direct_lighting_integrator.cuh"
#include "../utils/cuda_defines.cuh"

struct PixelData{
    float3 radiance;
    float3 albedo;
    float3 normal;
    float depth;
};

struct CurrentLight
{
    Ray ray;
    float3 weight;
    bool inside;
    int depth;
};

struct WhittedIntegrator
{

    __device__ static PixelData lighting(
        const CudaScene &scene,
        const Ray &primaryRay,
        const float tMin,
        const float tMax,
        curandState *rng);

    __device__ static float3 toneMap(const float3 &c);

    __device__ static float3 getSkyColor(const Ray &p_ray);
};