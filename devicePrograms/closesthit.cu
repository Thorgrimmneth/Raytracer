#include "../src/renderingGPU/optix/optix_payload.h"
#include "../src/renderingGPU/utils/packing.h"
#include "../src/renderingGPU/utils/objectType.h"
#include "../src/renderingGPU/optix/optix_sbt_manager.h"
#include "../src/renderingGPU/utils/op.cuh"
#include <optix.h>
#include <optix_device.h>

extern "C" __global__ void __closesthit__radiance()
{
    const HitData *data = reinterpret_cast<const HitData *>(optixGetSbtDataPointer());

    Payload *payload = reinterpret_cast<Payload *>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));

    const uint primID = optixGetPrimitiveIndex();

    const uint3 tri = data->triangles[primID];

    const float3 v0 = data->vertices[tri.x];

    const float3 v1 = data->vertices[tri.y];

    const float3 v2 = data->vertices[tri.z];

    const float2 bc = optixGetTriangleBarycentrics();

    const float b1 = bc.x;
    const float b2 = bc.y;
    const float b0 = 1.f - b1 - b2;

    const float3 N = normalize(cross(v1 - v0, v2 - v0));

    payload->hit = 1;

    payload->t = optixGetRayTmax();

    payload->position = optixGetWorldRayOrigin() + payload->t * optixGetWorldRayDirection();

    payload->normal = N;

    payload->objectIndex = primID;

    payload->materialIndex = data->materialIndex;

    payload->objectType = HIT_TRIANGLE_MESH;
}