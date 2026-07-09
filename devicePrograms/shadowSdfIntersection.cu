#include "../src/renderingGPU/optix/optix_payload.h"
#include "../src/renderingGPU/optix/optix_sbt_manager.h"
#include "../src/renderingGPU/utils/op.cuh"
#include "../src/renderingGPU/utils/packing.h"

#include <optix.h>
#include <optix_device.h>

#include "launch_shadow_params.cuh"

extern "C" {
__constant__ LaunchShadowParams params;
}

extern "C" __global__ void __intersection__sdf__shadow()
{
    const HitData *data = reinterpret_cast<const HitData *>(optixGetSbtDataPointer());

    const uint primIdx = optixGetPrimitiveIndex();

    const SDF &sdf = data->sdf.sdfs[primIdx];

    float3 rayOrigin = optixGetWorldRayOrigin();
    float3 rayDirection = optixGetWorldRayDirection();

    float tMin = optixGetRayTmin();
    float tMax = optixGetRayTmax();

    float tHit;
    float relaxationFactor = 1.5f;
    float oldDistance = 20000.f;
    float oldStep = 0.f;
    for (int i = 0; i < 64; i++)
    {
        float3 point = rayOrigin + tMin * rayDirection;
        float distance = sdf.sdf(point);

        if (distance < 1e-4f)
        {
            tHit = tMin;
            optixReportIntersection(tHit, 0);
            return;
        }

        if (relaxationFactor > 1.0f && fabs(distance) + fabs(oldDistance) < oldStep)
        {
            tMin += oldStep * (1.f - relaxationFactor);
            relaxationFactor = 1.0f;
            oldDistance = distance;
            oldStep = 0.f;
            continue;
        }

        oldDistance = distance;
        oldStep = distance * relaxationFactor;
        tMin += oldStep;

        if (tMin > tMax)
            break;
    }
}