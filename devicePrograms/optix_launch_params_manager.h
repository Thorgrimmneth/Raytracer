#pragma once

#include "../src/renderingGPU/raytracingUtils/ray.cuh"
#include "launch_params.cuh"
#include <vector>

class OptixLaunchParamsManager
{
  public:
    void create();

    void destroy();

    LaunchParams params = {};

    CUdeviceptr d_params = 0;
    std::vector<uchar4> downloadFramebuffer() const;

  private:
    uchar4 *framebuffer = nullptr;
};