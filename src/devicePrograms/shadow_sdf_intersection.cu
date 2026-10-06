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

    float tMin = optixGetRayTmin();
    float tMax = optixGetRayTmax();

    if (sdf.type == SDFType::SphereAnalytic)
    {
        if (!sdf.sphere.intersect(sdf.translation, rayOrigin, rayDirection, tMin, tMax))
            return;
        optixReportIntersection(tMin, 0);
        return;
    }

    float3 rayOriginLocal = transform(sdf.rotation, rayOrigin - sdf.translation);
    float3 rayDirectionLocal = transform(sdf.rotation, rayDirection);
    if (!intersectAABB(rayOriginLocal, rayDirectionLocal, sdf.aabb, tMin, tMax))

        return;

    float epsilon = 1e-4f;
    if(sdf.type == SDFType::SignedGrid){
        epsilon = fminf(0.001, 0.1275f * (2.0f / ((float)sdf.signedGrid.resolution - 1.f)));
    }
    for (int i = 0; i < 256; i++)
    {
        float3 p = rayOriginLocal + tMin * rayDirectionLocal;

        float d = sdf.sdf(p);
        if (d < epsilon)
        {
            optixReportIntersection(tMin, 0);
            return;
        }
        if (d > 20000.f)
            return;
        tMin += d;

        if (tMin > tMax)
            break;
    }
}