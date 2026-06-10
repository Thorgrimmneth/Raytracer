#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <stdint.h>

struct LaunchParams
{
    uchar4* framebuffer;

    uint32_t width;
    uint32_t height;

    OptixTraversableHandle traversable;
};