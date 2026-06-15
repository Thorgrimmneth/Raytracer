#pragma once

#include "../utils/constant.cuh"

#include "../lights/light.cuh"
#include "../lights/light_selection_utils.cuh"

#include "../scene/scene.cuh"

#include "../../../devicePrograms/launch_params.cuh"

__global__ void shadeWavefrontKernel(CudaScene scene, float3 *origins, float3 *directions, float3 *throughput, float3 *radiance, int *pixelIndices, RNG *rng,
                                     bool *isInside, bool *lastBounceWasDelta, float *lastBsdfPdf, float3 *positions, float3 *normals, int *materialIndices,
                                     int *hitMask, const int *activeQueue, int activeCount, int *nextActiveQueue,
                                     int *nextActiveCount, bool safeSun, uint depth);