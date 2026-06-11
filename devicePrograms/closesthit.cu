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

    const float2 bc = optixGetTriangleBarycentrics();

    const float3& n0 = data->normals[tri.x];
    const float3& n1 = data->normals[tri.y];
    const float3& n2 = data->normals[tri.z];
    
    const float3 N = (1 - bc.x - bc.y) * n0 + bc.x * n1 + bc.y * n2;

    payload->hit = 1;

    payload->t = optixGetRayTmax();

    payload->position = optixGetWorldRayOrigin() + payload->t * optixGetWorldRayDirection();

    payload->normal = N;

    payload->objectIndex = primID;

    payload->materialIndex = data->materialIndex;

    payload->objectType = HIT_TRIANGLE_MESH;
}