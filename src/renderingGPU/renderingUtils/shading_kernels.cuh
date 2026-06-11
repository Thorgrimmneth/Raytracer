#pragma once

#include "../utils/constant.cuh"

#include "../lights/light.cuh"
#include "../lights/light_selection_utils.cuh"

#include "../scene/scene.cuh"

#include "../../../devicePrograms/launch_params.cuh"

__global__ void shadeWavefrontKernel(CudaScene scene, Ray *rays, float3 *throughput, float3 *radiance, int *pixelIndices, RNG *rng,
                                     bool *isInside, bool *lastBounceWasDelta, float *lastBsdfPdf, OptixHit *hits,
                                     int *hitMask, const int *activeQueue, int activeCount, int *nextActiveQueue,
                                     int *nextActiveCount, bool safeSun, uint depth);