#pragma once

#include "../scene/scene.cuh"

#include "../utils/cuda_defines.cuh"
#include "../utils/rng.cuh"
#include "../utils/constant.cuh"
#include "../utils/macro.cuh"

#include "../raytracingUtils/ray.cuh"
#include "../raytracingUtils/hitrecord.cuh"

#include "../lights/light_selection_utils.cuh"

struct PathtracerIntegrator
{

    DEVICE static float3 lighting(
        const CudaScene &scene,
        const float3 &origin, const float3 &direction,
        const float tMin,
        const float tMax,
        RNG &rng);

    DEVICE static float3 getSkyColor(const float3 &origin, const float3 &direction, bool safeSun);

    
};