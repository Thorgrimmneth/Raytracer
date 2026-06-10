#include "../src/renderingGPU/optix/optix_payload.h"
#include "../src/renderingGPU/utils/op.cuh"
#include "../src/renderingGPU/utils/packing.h"
#include <optix.h>
#include <optix_device.h>

#include "launch_params.cuh"

extern "C" {
__constant__ LaunchParams params;
}

extern "C" __global__ void __raygen__render()
{
    uint3 idx = optixGetLaunchIndex();

    uint32_t pixel = idx.y * params.width + idx.x;

    Payload payload;

    payload.color = make_uchar4(0, 0, 0, 255);

    uint32_t p0;
    uint32_t p1;

    packPointer(&payload, p0, p1);

    float3 camPos = make_float3(8.f, 2.f, 3.f);
    float3 camTarget = make_float3(0.f, 0.f, 0.f);

    float3 forward = normalize(camTarget - camPos);

    float3 right = normalize(cross(forward, make_float3(0.f, 1.f, 0.f)));

    float3 up = cross(right, forward);

    float2 screen;

    screen.x = ((idx.x + 0.5f) / params.width) * 2.f - 1.f;

    screen.y = ((idx.y + 0.5f) / params.height) * 2.f - 1.f;

    float aspect = (float)params.width / (float)params.height;

    float fov = 45.f;

    float scale = tanf(fov * 0.5f * 3.14159265f / 180.f);

    float3 direction = normalize(forward + screen.x * aspect * scale * right + screen.y * scale * up);

    optixTrace(params.traversable, camPos, direction, 0.0f, 1e20f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
               0, 1, 0, p0, p1);

    params.framebuffer[pixel] = payload.color;
}