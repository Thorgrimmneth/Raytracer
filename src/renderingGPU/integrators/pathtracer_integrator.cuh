#pragma once

#include "../scene.cuh"
#include "../raytracingUtils/ray.cuh"
#include "../raytracingUtils/hitrecord.cuh"
#include "../utils/cuda_defines.cuh"
#include "../utils/rng.cuh"
#include "../utils/constant.cuh"
#include "../utils/macro.cuh"

struct PathtracerIntegrator
{

    DEVICE static float3 lighting(
        const CudaScene &scene,
        const Ray &primaryRay,
        const float tMin,
        const float tMax,
        RNG *rng);

    DEVICE
    void intersect(const CudaScene &scene, const Ray &ray, const float tMin, const float tMax, HitRecord &hit);

    DEVICE static float3 getSkyColor(const Ray &p_ray);

    
};