#include "../renderingGPU/optix/optix_payload.h"
#include "../renderingGPU/utils/packing.h"
#include "launch_radiance_params.cuh"
#include <optix.h>
#include <optix_device.h>

extern "C" {
__constant__ LaunchRadianceParams params;
}

extern "C" __global__ void __anyhit__shadow()
{
    ShadowPayload *payload = reinterpret_cast<ShadowPayload *>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));
    if (payload->depth >= 5) // Limit the depth of shadow rays to avoid infinite recursion
    {
        optixTerminateRay();
    }
    
    uint instanceIndex = optixGetInstanceId();
    if (!params.lightContext.meshInstances || instanceIndex >= params.nbMeshInstances)
    {
        optixIgnoreIntersection();
        return;
    }
    
    const Material &mat = params.lightContext.materials[params.lightContext.meshInstances[instanceIndex].materialIndex];
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