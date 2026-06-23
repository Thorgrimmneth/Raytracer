#pragma once

#include "../utils/constant.cuh"

#include "../lights/light.cuh"
#include "../lights/light_selection_utils.cuh"

#include "../scene/scene.cuh"

#include "../../../devicePrograms/launch_params.cuh"

__global__ void shadeMissKernel(float3 *origins, float3 *directions, float3 *throughput, float3 *accumBuffer,
                                int *pixelIndices, int missCount, bool safeSun);

__global__ void shadeLambertKernel(CudaScene scene, float3 *directions, float3 *p_throughput, RNG *p_rng,
                                   float3 *p_hitPositions, float3 *p_hitNormals, int *p_hitMaterialIndices,
                                   int *pixelIndices, float3 *nextOrigins, float3 *nextDirections,
                                   float3 *nextThroughput, int *nextPixelIndices, bool *nextLastBounceWasDelta,
                                   float *nextLastBsdfPdf, bool *nextIsInside, RNG *nextRNG, int *nextActiveCount,
                                   int materialCount, uint depth);

__global__ void shadeMetalKernel(CudaScene scene, float3 *origins, float3 *directions, float3 *throughputs, RNG *rngs,
                                 float3 *hitPositions, float3 *hitNormals, int *hitMaterialIndices, int *pixelIndices,
                                 float3 *accumBuffer, float3 *nextOrigins, float3 *nextDirections,
                                 float3 *nextThroughput, int *nextPixelIndices, bool *nextLastBounceWasDelta,
                                 float *nextLastBsdfPdf, bool *nextIsInside, RNG *nextRng, int *nextActiveCount,
                                 int activeCount, uint depth);

__global__ void shadePlasticNEEKernel(CudaScene scene, float3 *directions, float3 *throughputs, RNG *rngs,
                                      float3 *hitPositions, float3 *hitNormals,
                                      int *hitMaterialIndices, int *pixelIndices, float3 *accumBuffer,
                                      int nbLights, float *lightWeights, int activeCount);

__global__ void shadePlasticKernel(Material *materials, float3 *directions, float3 *throughputs, RNG *rngs,
                                   float3 *hitPositions, float3 *hitNormals, int *hitMaterialIndices,
                                   int *pixelIndices, float3 *accumBuffer, float3 *nextOrigins,
                                   float3 *nextDirections, float3 *nextThroughput, int *nextPixelIndices,
                                   bool *nextLastBounceWasDelta, float *nextLastBsdfPdf, bool *nextIsInside,
                                   RNG *nextRng, int *nextActiveCount, int activeCount, uint depth);

__global__ void shadeMirrorKernel(CudaScene scene, float3 *directions, float3 *throughputs, RNG *rngs,
                                  float3 *hitPositions, float3 *hitNormals, int *hitMaterialIndices,
                                  int *pixelIndices, float3 *accumBuffer, float3 *nextOrigins,
                                  float3 *nextDirections, float3 *nextThroughput, int *nextPixelIndices,
                                  bool *nextLastBounceWasDelta, float *nextLastBsdfPdf, bool *nextIsInside,
                                  RNG *nextRng, int *nextActiveCount, int activeCount, uint depth);

__global__ void shadeTransparentKernel(CudaScene scene, float3 *directions, float3 *throughputs, RNG *rngs,
                                     bool *isInsides, float3 *hitPositions, float3 *hitNormals,
                                        int *hitMaterialIndices, int *pixelIndices, float3 *accumBuffer,
                                       float3 *nextOrigins, float3 *nextDirections, float3 *nextThroughput,
                                       int *nextPixelIndices, bool *nextLastBounceWasDelta, float *nextLastBsdfPdf,
                                       bool *nextIsInside, RNG *nextRng, int *nextActiveCount, int activeCount,
                                       uint depth);

__global__ void shadeEmissiveKernel(CudaScene scene, float3 *origins, float3 *directions,
                                    float3 *throughputs, bool *lastBounceWasDelta, float *lastBsdfPdf,
                                    int *hitMaterialIndices, int *pixelIndices, float3 *accumBuffer,
                                    int activeCount);

// Fonctions helpers inlinées
__forceinline__ __device__ float pow5(float x)
{
    float x2 = x * x;
    return x2 * x2 * x;
}

__forceinline__ __device__ void getTangentFrame(float3 normal, float3 &T, float3 &B)
{
    float nx = normal.x;
    float ny = normal.y;
    float sign = copysignf(1.0f, normal.z);
    float a = -1.0f / (sign + normal.z);
    float b = nx * ny * a;
    T = make_float3(1.0f + sign * nx * nx * a, sign * b, -sign * nx);
    B = make_float3(b, sign + ny * ny * a, -ny);
}

__forceinline__ __device__ float3 sampleGGX(float u1, float u2, float Vx, float Vy, float Vz)
{
    // GGX sampling Disney/Burley
    float r = sqrtf(u1);
    float phi = 2.f * M_PIf * u2;
    float sinPhi, cosPhi;
    sincosf(phi, &sinPhi, &cosPhi);

    float t1 = r * cosPhi;
    float t2 = r * sinPhi;
    float s = 0.5f * (1.f + Vz);
    float t1S = t1 * t1;
    t2 = (1.f - s) * sqrtf(fmaxf(0.f, 1.f - t1S)) + s * t2;
    float t2S = t2 * t2;
    float lenNh = sqrtf(fmaxf(1e-8f, t1S + t2S + (1.f - t1S - t2S)));
    return make_float3(t1 / lenNh, t2 / lenNh, sqrtf(fmaxf(0.f, 1.f - t1S - t2S)) / lenNh);
}

DEVICE inline int enqueuePath(int *nextActiveCount, float3 *nextOrigins, float3 *nextDirections, float3 *nextThroughput,
                              float3 *nextRadiance, int *nextPixelIndices, bool *nextLastBounceWasDelta,
                              float *nextLastBsdfPdf, bool *nextIsInside, RNG *nextRNG, const float3 &origin,
                              const float3 &direction, const float3 &throughput, const float3 &radiance, int pixelIndex,
                              bool lastBounceWasDelta, float lastBsdfPdf, bool isInside, const RNG &rng)
{
    int dst = atomicAdd(nextActiveCount, 1);

    nextOrigins[dst] = origin;
    nextDirections[dst] = direction;
    nextThroughput[dst] = throughput;
    nextRadiance[dst] = radiance;

    nextPixelIndices[dst] = pixelIndex;

    nextLastBounceWasDelta[dst] = lastBounceWasDelta;
    nextLastBsdfPdf[dst] = lastBsdfPdf;
    nextIsInside[dst] = isInside;

    nextRNG[dst] = rng;

    return dst;
}