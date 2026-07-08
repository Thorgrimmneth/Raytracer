#include "../src/renderingGPU/optix/optix_payload.h"
#include "../src/renderingGPU/optix/optix_sbt_manager.h"
#include "../src/renderingGPU/utils/objectType.h"
#include "../src/renderingGPU/utils/op.cuh"
#include "../src/renderingGPU/utils/packing.h"
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

    const SDF& sdf = data->sdf.sdfs[primIdx];

    float h = 1e-4f;
    float3 p = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    float nx = sdf.sdf(p + make_float3(h, 0, 0)) - sdf.sdf(p - make_float3(h, 0, 0));
    float ny = sdf.sdf(p + make_float3(0, h, 0)) - sdf.sdf(p - make_float3(0, h, 0));
    float nz = sdf.sdf(p + make_float3(0, 0, h)) - sdf.sdf(p - make_float3(0, 0, h));
    float3 N = normalize(make_float3(nx, ny, nz));

    if (dot(N, -optixGetWorldRayDirection()) < 0.0f)
        N = -N;

    payload->hit = 1;

    payload->t = optixGetRayTmax();

    payload->position = optixGetWorldRayOrigin() + payload->t * optixGetWorldRayDirection();

    payload->normal = N;

    payload->objectIndex = primIdx;

    // Get material index from mesh instance data
    uint instanceIndex = optixGetInstanceId();

    if (params.meshInstances && instanceIndex < params.nbMeshInstances)
    {
        payload->materialIndex = params.meshInstances[instanceIndex].materialIndex;
    }
    else
    {
        payload->materialIndex = 0; // Fallback to SBT data
    }

    payload->objectType = HIT_SDF;
}