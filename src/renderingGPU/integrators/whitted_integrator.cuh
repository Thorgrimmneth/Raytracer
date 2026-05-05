#pragma once

#include "../scene.cuh"
#include "../raytracingUtils/ray.cuh"
#include "../raytracingUtils/hitrecord.cuh"
#include "direct_lighting_integrator.cuh"
#include "../utils/cuda_defines.cuh"

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