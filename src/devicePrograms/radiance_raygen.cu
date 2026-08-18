#include "../renderingGPU/optix/optix_payload.h"
#include "../renderingGPU/optix/optix_ray_type.h"
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
    const uint qid = optixGetLaunchIndex().x;

    if (qid >= params.active_count)
        return;

    float3 origin = params.origins[qid];
    float3 direction = params.directions[qid];

    bool lastBounceWasDelta = params.lastBounceWasDelta ? params.lastBounceWasDelta[qid] : false;

    bool isInside = params.isInside ? params.isInside[qid] : false;

    for (int bounce = 0; bounce < params.maxBounces; ++bounce)
    {
        Payload payload{};
        payload.depth = bounce;

        payload.hit = 0;
        payload.contribution = make_float3(0.f);

        payload.lastBounceWasDelta = lastBounceWasDelta;

        payload.isInside = isInside;

        uint32_t p0;
        uint32_t p1;

        packPointer(&payload, p0, p1);

        optixTrace(params.traversable,

                   origin, direction,

                   1e-3f, 20000.f, 0.f,

                   OptixVisibilityMask(255),

                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,

                   RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE,

                   p0, p1);

        // ========================================================
        // MISS
        // ========================================================

        if (!payload.hit)
        {
            params.accum_buffer[qid] += params.throughputs[qid] * payload.contribution;

            break;
        }

        // ========================================================
        // NEE / EMISSION
        // ========================================================

        params.accum_buffer[qid] += payload.contribution;

        // ========================================================
        // TERMINATE PATH
        // ========================================================

        if (payload.bsdfPdf <= 0.f)
            break;

        // ========================================================
        // NEXT BOUNCE
        // ========================================================

        // Offset ray origin away from surface, accounting for being inside transparent material
        direction = payload.bsdfDir;

        float side = dot(payload.bsdfDir, payload.normal) > 0.f ? 1.f : -1.f;

        origin = payload.position + payload.normal * (side * 1e-3f);

        direction = payload.bsdfDir;

        lastBounceWasDelta = payload.lastBounceWasDelta;

        isInside = payload.isInside;

        // ========================================================
        // RUSSIAN ROULETTE
        // ========================================================

        if (bounce >= 3 && params.rngs)
        {
            float3 throughput = params.throughputs[qid];
            float survivalProb = fminf(fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)), 0.95f);

            if (params.rngs[qid].nextFloat() > survivalProb)
                break;

            params.throughputs[qid] = throughput / fmaxf(survivalProb, 1e-3f);
        }
    }

    // Store final path state if you still need it outside
    params.origins[qid] = origin;
    params.directions[qid] = normalize(direction);

    params.lastBounceWasDelta[qid] = lastBounceWasDelta;

    params.isInside[qid] = isInside;
}