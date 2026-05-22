#pragma once

#include "wavefront_state.cuh"

#include "../utils/constant.cuh"

#include "../lights/light.cuh"

#include "../scene/scene.cuh"

__global__ void shadeWavefrontKernel(CudaScene scene, WavefrontState *states, HitRecord *hits, int *hitMask,
                                     const int *activeQueue, int activeCount, int *nextActiveQueue,
                                     int *nextActiveCount, bool safeSun);