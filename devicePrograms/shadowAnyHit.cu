#include "../src/renderingGPU/optix/optix_payload.h"
#include <optix.h>
#include <optix_device.h>
#include "launch_shadow_params.cuh"
#include "../src/renderingGPU/utils/packing.h"

extern "C" {
__constant__ LaunchShadowParams params;
}

extern "C" __global__
void __anyhit__shadow()
{
    ShadowPayload* payload =
        reinterpret_cast<ShadowPayload*>(
            unpackPointer(
                optixGetPayload_0(),
                optixGetPayload_1()));

    Material& mat =
        params.materials[
            params.meshInstances[
                optixGetInstanceId()
            ].materialIndex];
    MaterialType type = mat.type();
    if(type != MaterialType::TRANSPARENT)
    {
        if(type != MaterialType::EMISSIVE)
        {
            payload->transmittance = make_float3(0.0f);
        }
        optixTerminateRay();
    }

    payload->transmittance *= mat.computeTransmission();

    optixIgnoreIntersection();
}