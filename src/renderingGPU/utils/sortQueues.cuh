#pragma once
#include "../materials/material.cuh"

struct MaterialRanges
{
    int offset[7];
    int count[7];
};

GLOBAL
void classifyPairs(Material *materials, int activeCount, MaterialType *keys, int *values, const int *hitMask,
                   const int *hitMaterialIndices);

GLOBAL
void reorderPaths(const int *permutation, const float3 *origins, const float3 *directions, const float3 *throughput,
                  const float3 *hitPositions, const float3 *hitNormals, const int *hitMaterialIndices,
                  const int *pixelIndices, const bool *lastBounceWasDelta, const float *lastBsdfPdf,
                  const bool *isInside, const RNG *rng, float3 *sortedOrigins, float3 *sortedDirections,
                  float3 *sortedThroughput, float3 *sortedHitPositions, float3 *sortedHitNormals,
                  int *sortedHitMaterialIndices, int *sortedPixelIndices, bool *sortedLastBounceWasDelta,
                  float *sortedLastBsdfPdf, bool *sortedIsInside, RNG *sortedRNG, int count);

__global__ void computeMaterialRanges(const MaterialType *keys, int activeCount, MaterialRanges *ranges);

__global__ void buildOctantKeys(const float3 *directions, int count, unsigned int *keys, int *values);