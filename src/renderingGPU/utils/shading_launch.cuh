#pragma once

#include "../materials/material.cuh"
#include "../scene/scene.cuh"
#include "shading_data.cuh"
#include "shadow_data.cuh"

dim3 block1D(256);

inline dim3 gridForCount(int count, int blockSize = 256) { return dim3((count + blockSize - 1) / blockSize); }

template <MaterialType T>
void launchShadeKernel(SortedRayQueue &in, RayQueue &out, CudaScene scene, float3 *accumBuffer, int offset,
                       int activeCount, ShadowRayQueue &shadowQueue, int *d_shadowCount, int depth);

template <>
void launchShadeKernel<MaterialType::MISS>(SortedRayQueue &in, RayQueue &out, CudaScene scene, float3 *accumBuffer,
                                           int offset, int activeCount, ShadowRayQueue &shadowQueue, int *d_shadowCount,
                                           int depth)
{
    shadeMissKernel<<<gridForCount(activeCount), block1D>>>(in.origins + offset, in.directions + offset,
                                                            in.throughputs + offset, accumBuffer,
                                                            in.pixelIndices + offset, activeCount, depth == 0);
}

template <>
void launchShadeKernel<MaterialType::LAMBERT>(SortedRayQueue &in, RayQueue &out, CudaScene scene, float3 *accumBuffer,
                                              int offset, int activeCount, ShadowRayQueue &shadowQueue,
                                              int *d_shadowCount, int depth)
{
    shadeLambertKernel<<<gridForCount(activeCount), block1D>>>(
        scene, in.directions + offset, in.throughputs + offset, in.rng + offset, in.hitPositions + offset,
        in.hitNormals + offset, in.hitMaterialIndices + offset, in.pixelIndices + offset, out.origins, out.directions,
        out.throughputs, out.pixelIndices, out.lastBounceWasDelta, out.lastBsdfPdf, out.isInside, out.rng,
        out.activeCount, activeCount, shadowQueue.origins, shadowQueue.directions, shadowQueue.contributions,
        shadowQueue.pixelIndices, shadowQueue.maxDistances, d_shadowCount, depth);
}

template <>
void launchShadeKernel<MaterialType::METAL>(SortedRayQueue &in, RayQueue &out, CudaScene scene, float3 *accumBuffer,
                                            int offset, int activeCount, ShadowRayQueue &shadowQueue,
                                            int *d_shadowCount, int depth)
{
    shadeMetalKernel<<<gridForCount(activeCount), block1D>>>(
        scene, in.directions + offset, in.throughputs + offset, in.rng + offset, in.hitPositions + offset,
        in.hitNormals + offset, in.hitMaterialIndices + offset, in.pixelIndices + offset, accumBuffer, out.origins,
        out.directions, out.throughputs, out.pixelIndices, out.lastBounceWasDelta, out.lastBsdfPdf, out.isInside,
        out.rng, out.activeCount, activeCount, shadowQueue.origins, shadowQueue.directions, shadowQueue.contributions,
        shadowQueue.pixelIndices, shadowQueue.maxDistances, d_shadowCount, depth);
}

template <>
void launchShadeKernel<MaterialType::PLASTIC>(SortedRayQueue &in, RayQueue &out, CudaScene scene, float3 *accumBuffer,
                                              int offset, int activeCount, ShadowRayQueue &shadowQueue,
                                              int *d_shadowCount, int depth)
{
    shadePlasticNEEKernel<<<gridForCount(activeCount), block1D>>>(
        scene, in.directions + offset, in.throughputs + offset, in.rng + offset, in.hitPositions + offset,
        in.hitNormals + offset, in.hitMaterialIndices + offset, in.pixelIndices + offset, accumBuffer, scene.nbLights,
        scene.lightCumulativeWeights, activeCount, shadowQueue.origins, shadowQueue.directions, shadowQueue.contributions,
        shadowQueue.pixelIndices, shadowQueue.maxDistances, d_shadowCount);

    shadePlasticKernel<<<gridForCount(activeCount), block1D>>>(
        scene.materials, in.directions + offset, in.throughputs + offset, in.rng + offset, in.hitPositions + offset,
        in.hitNormals + offset, in.hitMaterialIndices + offset, in.pixelIndices + offset, accumBuffer, out.origins,
        out.directions, out.throughputs, out.pixelIndices, out.lastBounceWasDelta, out.lastBsdfPdf, out.isInside,
        out.rng, out.activeCount, activeCount, depth);
}

template <>
void launchShadeKernel<MaterialType::MIRROR>(SortedRayQueue &in, RayQueue &out, CudaScene scene, float3 *accumBuffer,
                                             int offset, int activeCount, ShadowRayQueue &shadowQueue,
                                             int *d_shadowCount, int depth)
{
    shadeMirrorKernel<<<gridForCount(activeCount), block1D>>>(
        scene, in.directions + offset, in.throughputs + offset, in.rng + offset, in.hitPositions + offset,
        in.hitNormals + offset, in.hitMaterialIndices + offset, in.pixelIndices + offset, accumBuffer, out.origins,
        out.directions, out.throughputs, out.pixelIndices, out.lastBounceWasDelta, out.lastBsdfPdf, out.isInside,
        out.rng, out.activeCount, activeCount, depth);
}

template <>
void launchShadeKernel<MaterialType::TRANSPARENT>(SortedRayQueue &in, RayQueue &out, CudaScene scene,
                                                  float3 *accumBuffer, int offset, int activeCount,
                                                  ShadowRayQueue &shadowQueue, int *d_shadowCount, int depth)
{
    shadeTransparentKernel<<<gridForCount(activeCount), block1D>>>(
        scene, in.directions + offset, in.throughputs + offset, in.rng + offset, in.isInside + offset,
        in.hitPositions + offset, in.hitNormals + offset, in.hitMaterialIndices + offset, in.pixelIndices + offset,
        accumBuffer, out.origins, out.directions, out.throughputs, out.pixelIndices, out.lastBounceWasDelta,
        out.lastBsdfPdf, out.isInside, out.rng, out.activeCount, activeCount, depth);
}

template <>
void launchShadeKernel<MaterialType::EMISSIVE>(SortedRayQueue &in, RayQueue &out, CudaScene scene, float3 *accumBuffer,
                                               int offset, int activeCount, ShadowRayQueue &shadowQueue,
                                               int *d_shadowCount, int depth)
{
    shadeEmissiveKernel<<<gridForCount(activeCount), block1D>>>(
        scene, in.origins + offset, in.directions + offset, in.throughputs + offset, in.lastBounceWasDelta + offset,
        in.lastBsdfPdf + offset, in.hitMaterialIndices + offset, in.pixelIndices + offset, accumBuffer, activeCount);
}

void launchShadeKernel(MaterialType type, SortedRayQueue &in, RayQueue &out, CudaScene scene, float3 *accumBuffer,
                       int offset, int activeCount, ShadowRayQueue &shadowQueue, int *d_shadowCount, int depth)
{
    switch (type)
    {
    case MaterialType::MISS:
        launchShadeKernel<MaterialType::MISS>(in, out, scene, accumBuffer, offset, activeCount, shadowQueue,
                                              d_shadowCount, depth);
        break;
    case MaterialType::LAMBERT:
        launchShadeKernel<MaterialType::LAMBERT>(in, out, scene, accumBuffer, offset, activeCount, shadowQueue,
                                                 d_shadowCount, depth);
        break;

    case MaterialType::METAL:
        launchShadeKernel<MaterialType::METAL>(in, out, scene, accumBuffer, offset, activeCount, shadowQueue,
                                               d_shadowCount, depth);
        break;

    case MaterialType::PLASTIC:
        launchShadeKernel<MaterialType::PLASTIC>(in, out, scene, accumBuffer, offset, activeCount, shadowQueue,
                                                 d_shadowCount, depth);
        break;
    case MaterialType::MIRROR:
        launchShadeKernel<MaterialType::MIRROR>(in, out, scene, accumBuffer, offset, activeCount, shadowQueue,
                                                d_shadowCount, depth);
        break;
    case MaterialType::TRANSPARENT:
        launchShadeKernel<MaterialType::TRANSPARENT>(in, out, scene, accumBuffer, offset, activeCount, shadowQueue,
                                                     d_shadowCount, depth);
        break;
    case MaterialType::EMISSIVE:
        launchShadeKernel<MaterialType::EMISSIVE>(in, out, scene, accumBuffer, offset, activeCount, shadowQueue,
                                                  d_shadowCount, depth);
        break;

    default:
        throw std::runtime_error("Unknown material type");
    }
}