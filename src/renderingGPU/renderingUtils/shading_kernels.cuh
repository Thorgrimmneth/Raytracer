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

__global__ void shadeLambertKernel(CudaScene scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                   float3 *p_radiance, RNG *p_rng, OptixHit *p_hits, bool *lastBounceWasDelta, const int *hitQueue, int hitCount,
                                   int *nextActiveQueue, int *nextActiveCount, uint depth);
__global__ void shadeMetalKernel(CudaScene scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                 float3 *p_radiance, RNG *p_rng, OptixHit *p_hits, bool *lastBounceWasDelta,const int *hitQueue, int hitCount,
                                 int *nextActiveQueue, int *nextActiveCount, uint depth);

__global__ void shadePlasticNEEKernel(CudaScene &scene, float4 *directions, float3 *p_throughput, float3 *p_radiance,
                                      RNG *p_rng, OptixHit *p_hits, const int *hitQueue, int hitCount);

__global__ void shadePlasticKernel(CudaScene &scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                        RNG *p_rng, OptixHit *p_hits, bool *lastBounceWasDelta,const int *hitQueue,
                                       int hitCount, int *nextActiveQueue, int *nextActiveCount, uint depth);

__global__ void shadeMirrorKernel(CudaScene scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                  RNG *p_rng, bool *p_lastBounceWasDelta, float *p_lastBsdfPdf, OptixHit *p_hits,
                                  const int *hitQueue, int hitCount, int *nextActiveQueue, int *nextActiveCount,
                                  uint depth);

__global__ void shadeTransparentKernel(CudaScene scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                       RNG *p_rng, bool *p_isInside, bool *p_lastBounceWasDelta, float *p_lastBsdfPdf,
                                       OptixHit *p_hits, const int *hitQueue, int hitCount, int *nextActiveQueue,
                                       int *nextActiveCount, uint depth);

__global__ void shadeEmissiveKernel(CudaScene &scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                    float3 *p_radiance, bool *p_lastBounceWasDelta, float *p_lastBsdfPdf,
                                    OptixHit *p_hits, int *hitMask, const int *activeQueue, int activeCount);