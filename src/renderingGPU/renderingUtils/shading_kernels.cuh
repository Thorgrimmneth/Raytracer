#pragma once

#include "../utils/constant.cuh"

#include "../lights/light.cuh"
#include "../lights/light_selection_utils.cuh"

#include "../scene/scene.cuh"

#include "../../../devicePrograms/launch_params.cuh"

__global__ void shadeWavefrontKernel(CudaScene &scene, float3 *origins, float4 *directions, float3 *throughput,
                                     float3 *radiance, int *pixelIndices, RNG *rng, bool *isInside,
                                     bool *lastBounceWasDelta, float *lastBsdfPdf, OptixHit *hits, int *hitMask,
                                     const int *activeQueue, int activeCount, int *nextActiveQueue,
                                     int *nextActiveCount, bool safeSun, uint depth);

__global__ void shadeMissKernel(float3 *origins, float4 *directions, float3 *throughput, float3 *radiance,
                                const int *missQueue, int missCount, bool safeSun);

__global__ void shadeHitKernel(CudaScene scene, float3 *origins, float4 *directions, float3 *p_throughput,
                               float3 *p_radiance, int *pixelIndices, RNG *p_rng, bool *p_isInside,
                               bool *p_lastBounceWasDelta, float *p_lastBsdfPdf, OptixHit *hits, const int *hitQueue,
                               int hitCount, int *nextActiveQueue, int *nextActiveCount, uint depth);