#include "../renderingGPU/materials/material.cuh"
#include "../renderingGPU/optix/optix_payload.h"
#include "../renderingGPU/optix/optix_sbt_manager.h"
#include "../renderingGPU/utils/op.cuh"
#include "../renderingGPU/utils/packing.h"
#include <optix.h>
#include <optix_device.h>

#include "launch_radiance_params.cuh"

extern "C" {
__constant__ LaunchRadianceParams params;
}

extern "C" __global__ void __intersection__sdf()
{
    const HitData *data = reinterpret_cast<const HitData *>(optixGetSbtDataPointer());

    const uint primIdx = optixGetPrimitiveIndex();
    const SDF &sdf = data->sdf.sdfs[primIdx];
    int materialIdx = sdf.getMaterialIndex();
    // si on est à l'intérieur d'un objet transparent, on applique la valeur absolue de la SDF pour éviter les
    // intersections négatives
    bool applyAbs = params.lightContext.materials[materialIdx].type() == MaterialType::TRANSPARENT;

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
    if(sdf.type == SDFType::SignedGrid)
    {
        epsilon = fminf(epsilon, 0.25f * (2.0f / (float)(sdf.signedGrid.resolution - 1)));
    }
    epsilon = 0.001f;
    for (int i = 0; i < 256; i++)
    {
        float3 p = rayOriginLocal + tMin * rayDirectionLocal;

        float d = sdf.sdf(p);
        if (applyAbs)
            d = fabsf(d);
        
        if (d < epsilon)
        {
            //Payload *payload = reinterpret_cast<Payload *>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));
            //payload->nbIter += i;
            optixReportIntersection(tMin, 0);
            return;
        }
        if (d > 20000.f)
            return;

        tMin += d * 0.5f;

        if (tMin > tMax)
            break;
    }
}