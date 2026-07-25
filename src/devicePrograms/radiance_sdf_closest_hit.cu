#include "../renderingGPU/optix/optix_payload.h"
#include "../renderingGPU/optix/optix_sbt_manager.h"
#include "../renderingGPU/utils/object_type.h"
#include "../renderingGPU/utils/op.cuh"
#include "../renderingGPU/utils/packing.h"
#include "launch_radiance_params.cuh"
#include <optix.h>
#include <optix_device.h>

extern "C" {
__constant__ LaunchRadianceParams params;
}

extern "C" __global__ void __closesthit__radiance__sdf()
{
    const HitData *data = reinterpret_cast<const HitData *>(optixGetSbtDataPointer());
    Payload *payload = reinterpret_cast<Payload *>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));

    const uint primIdx = optixGetPrimitiveIndex();

    const SDF &sdf = data->sdf.sdfs[primIdx];

    float3 p = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    float3 NWorld;
    if (sdf.type == SDFType::SphereAnalytic)
    {
        NWorld = sdf.sphere.getNormal(p, sdf.translation);
    }
    else
    {
        float3 pLocal = transform(sdf.rotation, p - sdf.translation);
        float3 N = sdf.getNormal(pLocal);

        NWorld = normalize(transform(sdf.rotation.transpose(), N));
    }

    payload->hit = 1;

    payload->t = optixGetRayTmax();

    payload->position = optixGetWorldRayOrigin() + payload->t * optixGetWorldRayDirection();

    payload->normal = NWorld;

    payload->objectIndex = primIdx;

    // Get material index from mesh instance data
    uint instanceIndex = optixGetInstanceId();

    payload->materialIndex = sdf.getMaterialIndex();

    payload->object_type = HIT_SDF;

    payload->objectIndex = 28;
}