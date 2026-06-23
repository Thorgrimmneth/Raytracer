#include "../src/renderingGPU/optix/optix_payload.h"
#include "../src/renderingGPU/utils/op.cuh"

#include "../src/renderingGPU/utils/packing.h"

#include <optix.h>
#include <optix_device.h>

#include "launch_params.cuh"

extern "C" {
__constant__ LaunchParams params;
}

extern "C" __global__ void __raygen__intersect()
{
    uint3 launchIndex = optixGetLaunchIndex();

    uint qid = launchIndex.x;

    if (qid >= params.activeCount)
        return;

    Payload payload;

    payload.hit = 0;

    uint32_t p0;
    uint32_t p1;

    packPointer(&payload, p0, p1);

    optixTrace(params.traversable, params.origins[qid], params.directions[qid], 0.001f, 1e20f, 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1);

    params.hitMask[qid] = payload.hit;

    if (payload.hit)
    {
        /*printf("Hit at index %d: position (%f, %f, %f), normal (%f, %f, %f), materialIndex %d\n", qid,
               payload.position.x, payload.position.y, payload.position.z,
               payload.normal.x, payload.normal.y, payload.normal.z,
               payload.materialIndex);*/
        // Store hit data in SoA format
        params.hitPositions[qid] = payload.position;
        params.hitNormals[qid] = payload.normal;
        params.hitMaterialIndices[qid] = payload.materialIndex;
    }
}