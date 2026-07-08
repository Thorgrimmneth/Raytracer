#include "../src/renderingGPU/objects/sdf.cuh"
#include "../src/renderingGPU/optix/optix_payload.h"
#include "../src/renderingGPU/optix/optix_sbt_manager.h"
#include "../src/renderingGPU/utils/packing.h"
#include "launch_shadow_params.cuh"
#include <optix.h>
#include <optix_device.h>

extern "C" {
__constant__ LaunchShadowParams params;
}

extern "C" __global__ void __anyhit__shadow__sdf()
{
    ShadowPayload *payload = reinterpret_cast<ShadowPayload *>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));
    if (payload->depth >= 5) // Limit the depth of shadow rays to avoid infinite recursion
    {
        optixTerminateRay();
    }
    const HitData *data = reinterpret_cast<const HitData *>(optixGetSbtDataPointer());
    const SDF &sdf = data->sdf.sdfs[optixGetPrimitiveIndex()];

    Material &mat = params.materials[sdf.getMaterialIndex()];
    MaterialType type = mat.type();
    if (type != MaterialType::TRANSPARENT)
    {
        if (type != MaterialType::EMISSIVE)
        {
            payload->transmittance = make_float3(0.f);
        }
        optixTerminateRay();
    }

    payload->transmittance *= mat.computeTransmission();
    payload->depth++;
    float t = fmaxf(payload->transmittance.x, fmaxf(payload->transmittance.y, payload->transmittance.z));

    if (t < 1e-3f)
    {
        payload->transmittance = make_float3(0.f);
        optixTerminateRay();
    }
    optixIgnoreIntersection();
}