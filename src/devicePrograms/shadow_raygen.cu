#include "../renderingGPU/optix/optix_payload.h"
#include "../renderingGPU/utils/op.cuh"

#include "../renderingGPU/utils/packing.h"

#include <optix.h>
#include <optix_device.h>

#include "launch_shadow_params.cuh"

extern "C" {
__constant__ LaunchShadowParams params;
}

extern "C" __global__ void __raygen__shadow()
{
    uint3 launchIndex = optixGetLaunchIndex();

    uint qid = launchIndex.x;

    if (qid >= params.activeCount)
        return;

    ShadowPayload payload;

    payload.transmittance = make_float3(1.f);
    payload.depth = 0;

    uint32_t p0;
    uint32_t p1;

    packPointer(&payload, p0, p1);

    optixTrace(params.traversable, params.origins[qid], params.directions[qid], 1e-3f, params.maxDistances[qid], 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1);

    params.transmittance[qid] = payload.transmittance;
}