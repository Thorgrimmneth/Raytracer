#include "../renderingGPU/utils/packing.h"
#include "../renderingGPU/optix/optix_payload.h"
#include <optix.h>
#include <optix_device.h>

extern "C" __global__ void __miss__radiance()
{
    Payload *payload = reinterpret_cast<Payload *>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));

    payload->hit = 0;
}