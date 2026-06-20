#pragma once

#include "../utils/constant.cuh"

#include "../lights/light.cuh"
#include "../lights/light_selection_utils.cuh"

#include "../scene/scene.cuh"

#include "../../../devicePrograms/launch_params.cuh"

__global__ void shadeWavefrontKernel(CudaScene &scene, float3 *origins, float4 *directions, float3 *throughput,
                                     float3 *radiance, int *pixelIndices, RNG *rng, bool *isInside,
                                     bool *lastBounceWasDelta, float *lastBsdfPdf, float4 *hitPositions,
                                     float4 *hitNormals, int *hitMaterialIndices, int *hitMask, const int *activeQueue,
                                     int activeCount, int *nextActiveQueue, int *nextActiveCount, bool safeSun,
                                     uint depth);

__global__ void shadeMissKernel(float4 *origins, float4 *directions, float4 *throughput, float3 *radiance,
                                const int *missQueue, int missCount, bool safeSun);

__global__ void shadeLambertKernel(CudaScene scene, float4 *origins, float4 *directions, float4 *p_throughput,
                                   float3 *p_radiance, RNG *p_rng, float4 *p_hitPositions, float4 *p_hitNormals,
                                   int *p_hitMaterialIndices, bool *lastBounceWasDelta, const int *hitQueue,
                                   int hitCount, int *nextActiveQueue, int *nextActiveCount, uint depth);
__global__ void shadeMetalKernel(CudaScene scene, float4 *origins, float4 *directions, float4 *p_throughput,
                                 float3 *p_radiance, RNG *p_rng, float4 *p_hitPositions, float4 *p_hitNormals,
                                 int *p_hitMaterialIndices, bool *lastBounceWasDelta, const int *hitQueue, int hitCount,
                                 int *nextActiveQueue, int *nextActiveCount, uint depth);

__global__ void shadePlasticNEEKernel(CudaScene &scene, float4 *directions, float4 *p_throughput, float3 *p_radiance,
                                      RNG *p_rng, float4 *p_hitPositions, float4 *p_hitNormals,
                                      int *p_hitMaterialIndices, const int *hitQueue, int hitCount, int nbLights);

__global__ void shadePlasticKernel(Material *materials, float4 *origins, float4 *directions, float4 *p_throughput,
                                   RNG *p_rng, float4 *p_hitPositions, float4 *p_hitNormals, int *p_hitMaterialIndices,
                                   bool *lastBounceWasDelta, const int *hitQueue, int hitCount, int *nextActiveQueue,
                                   int *nextActiveCount, uint depth);

__global__ void shadeMirrorKernel(CudaScene scene, float4 *origins, float4 *directions, float4 *p_throughput, RNG *p_rng,
                                 bool *p_lastBounceWasDelta, float *p_lastBsdfPdf, float4 *p_hitPositions,
                                 float4 *p_hitNormals, int *p_hitMaterialIndices, const int *hitQueue, int hitCount,
                                 int *nextActiveQueue, int *nextActiveCount, uint depth);

__global__ void shadeTransparentKernel(CudaScene scene, float4 *origins, float4 *directions, float4 *p_throughput,
                                       RNG *p_rng, bool *p_isInside, bool *p_lastBounceWasDelta, float *p_lastBsdfPdf,
                                       float4 *p_hitPositions, float4 *p_hitNormals, int *p_hitMaterialIndices,
                                       const int *hitQueue, int hitCount, int *nextActiveQueue, int *nextActiveCount,
                                       uint depth);

__global__ void shadeEmissiveKernel(CudaScene &scene, float4 *origins, float4 *directions, float4 *p_throughput,
                                    float3 *p_radiance, bool *p_lastBounceWasDelta, float *p_lastBsdfPdf,
                                    int *p_hitMaterialIndices, int *hitMask, const int *activeQueue, int activeCount);

// Fonctions helpers inlinées
__forceinline__ __device__ float pow5(float x)
{
    float x2 = x * x;
    return x2 * x2 * x;
}

__forceinline__ __device__ void getTangentFrame(float3 normal, float3 &T, float3 &B)
{
    float sign = copysignf(1.0f, normal.z);
    float a = -1.0f / (sign + normal.z);
    float b = normal.x * normal.y * a;
    T = make_float3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    B = make_float3(b, sign + normal.y * normal.y * a, -normal.y);
}

__forceinline__ __device__ float3 sampleGGX(float u1, float u2, float Vx, float Vy, float Vz)
{
    // GGX sampling Disney/Burley
    float r = sqrtf(u1);
    float phi = 2.f * M_PIf * u2;
    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    float t1 = r * cosPhi;
    float t2 = r * sinPhi;
    float s = 0.5f * (1.f + Vz);
    t2 = (1.f - s) * sqrtf(fmaxf(0.f, 1.f - t1 * t1)) + s * t2;

    float lenNh = sqrtf(fmaxf(1e-8f, t1 * t1 + t2 * t2 + (1.f - t1 * t1 - t2 * t2)));
    return make_float3(t1 / lenNh, t2 / lenNh, sqrtf(fmaxf(0.f, 1.f - t1 * t1 - t2 * t2)) / lenNh);
}