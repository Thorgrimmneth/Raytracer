#include "../renderingGPU/optix/optix_payload.h"
#include "../renderingGPU/utils/op.cuh"

#include "../renderingGPU/utils/packing.h"

#include <optix.h>
#include <optix_device.h>

#include "launch_radiance_params.cuh"

extern "C" {
__constant__ LaunchRadianceParams params;
}

extern "C" __global__ void __raygen__radiance()
{
    uint3 launchIndex = optixGetLaunchIndex();

    uint qid = launchIndex.x;

    if (qid >= params.active_count)
        return;

    Payload payload;

    payload.hit = 0;

    uint32_t p0;
    uint32_t p1;

    packPointer(&payload, p0, p1);
    optixTraverse(params.traversable, params.origins[qid], params.directions[qid], 0.001f, 1e20f, 0.0f,
                  OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0, p1);
    optixReorder();
    optixInvoke(p0, p1);
    //optixTrace(params.traversable, params.origins[qid], params.directions[qid], 0.001f, 1e20f, 0.0f,
    //   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0, p1);

    params.hit_buffers.mask[qid] = payload.hit;

    if (payload.hit)
    {
        /*printf("Hit at index %d: position (%f, %f, %f), normal (%f, %f, %f), materialIndex %d\n", qid,
               payload.position.x, payload.position.y, payload.position.z,
               payload.normal.x, payload.normal.y, payload.normal.z,
               payload.materialIndex);*/
        // Store hit data in SoA format
        params.hit_buffers.positions[qid] = payload.position;
        params.hit_buffers.normals[qid] = payload.normal;
        params.hit_buffers.materialIndices[qid] = payload.materialIndex;
        params.hit_buffers.distances[qid] = payload.t;
        params.hit_buffers.types[qid] = payload.object_type;
        params.hit_buffers.objectIndices[qid] = payload.objectIndex;
    }
}