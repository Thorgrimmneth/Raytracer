#pragma once

#include "launch_params.cuh"
#include <vector>

class OptixLaunchParamsManager
{
public:

    void create(
        uint32_t width,
        uint32_t height
    );

    void destroy();

    LaunchParams params = {};

    CUdeviceptr d_params = 0;
    std::vector<uchar4> downloadFramebuffer() const;
private:

    uchar4* framebuffer = nullptr;
};