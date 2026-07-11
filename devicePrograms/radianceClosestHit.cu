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

extern "C" __global__ void __closesthit__radiance()
{
    const HitData *data = reinterpret_cast<const HitData *>(optixGetSbtDataPointer());
    Payload *payload = reinterpret_cast<Payload *>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));

    const uint primID = optixGetPrimitiveIndex();

    const uint3 tri = data->mesh.triangles[primID];

    const float2 bc = optixGetTriangleBarycentrics();

    const float3 &n0 = data->mesh.normals[tri.x];
    const float3 &n1 = data->mesh.normals[tri.y];
    const float3 &n2 = data->mesh.normals[tri.z];

    float3 N = normalize((1 - bc.x - bc.y) * n0 + bc.x * n1 + bc.y * n2);

    if(dot(N, optixGetWorldRayDirection()) > 0.f)
    {
        N = -N;
    }
    payload->hit = 1;

    payload->t = optixGetRayTmax();

    payload->position = optixGetWorldRayOrigin() + payload->t * optixGetWorldRayDirection();

    payload->normal = N;

    payload->objectIndex = primID;

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

    payload->objectType = HIT_TRIANGLE_MESH;
}