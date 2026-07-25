#include "../renderingGPU/optix/optix_payload.h"
#include "../renderingGPU/optix/optix_sbt_manager.h"
#include "../renderingGPU/utils/op.cuh"
#include "../renderingGPU/utils/packing.h"

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
    float3 rayOriginLocal = transform(sdf.rotation, rayOrigin - sdf.translation);
    float3 rayDirectionLocal = transform(sdf.rotation, rayDirection);

    float tMin = optixGetRayTmin();
    float tMax = optixGetRayTmax();
    if(!intersectAABB(rayOriginLocal, rayDirectionLocal, sdf.aabb, tMin, tMax))
         return;

    for (int i = 0; i < 128; i++)
    {
        float3 p = rayOriginLocal + tMin * rayDirectionLocal;

        float d = sdf.sdf(p);
        if (d < 1e-4f)
        {
            optixReportIntersection(tMin, 0);
            return;
        }
        if(d > 1e20f)
            return;
        tMin += d;

        if (tMin > tMax)
            break;
    }
}