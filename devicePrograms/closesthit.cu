#include "../src/renderingGPU/optix/optix_payload.h"
#include "../src/renderingGPU/utils/packing.h"
#include <optix.h>
#include <optix_device.h>

extern "C" __global__ void __closesthit__radiance()
{
    Payload *payload = reinterpret_cast<Payload *>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));

    payload->color = make_uchar4(255, 0, 0, 255);
}