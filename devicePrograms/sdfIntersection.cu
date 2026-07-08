#include "../src/renderingGPU/optix/optix_payload.h"
#include "../src/renderingGPU/utils/op.cuh"
#include "../src/renderingGPU/optix/optix_sbt_manager.h"
#include "../src/renderingGPU/utils/packing.h"

#include <optix.h>
#include <optix_device.h>

#include "launch_radiance_params.cuh"

extern "C" {
__constant__ LaunchRadianceParams params;
}

extern "C" __global__ void __intersection__sdf()
{
    const HitData* data =
        reinterpret_cast<const HitData*>(optixGetSbtDataPointer());

    const uint primIdx = optixGetPrimitiveIndex();

    const SDF& sdf = data->sdf.sdfs[primIdx];

    float3 rayOrigin = optixGetWorldRayOrigin();
    float3 rayDirection = optixGetWorldRayDirection();

    float tMin = optixGetRayTmin();
    float tMax = optixGetRayTmax();

    float tHit;
    for(int i = 0; i < 64; i++)
    {
        float3 point = rayOrigin + tMin * rayDirection;
        float distance = sdf.sdf(point);

        if (distance < 1e-4f)
        {
            tHit = tMin;
            optixReportIntersection(tHit, 0);
            return;
        }

        tMin += distance;

        if (tMin > tMax)
            break;
    }
}