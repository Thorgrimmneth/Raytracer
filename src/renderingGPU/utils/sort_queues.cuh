#pragma once
#include "../materials/material.cuh"
#include "shading_data.cuh"

struct MaterialRanges
{
    int offset[MATERIAL_TYPE_COUNT];
    int count[MATERIAL_TYPE_COUNT];
};

GLOBAL
void classifyPairs(Material *materials, int activeCount, int *keys, int *values, const int *hitMask,
                   const int *hitMaterialIndices);

GLOBAL
void reorderPaths(const int *permutation, RayQueue current, SortedRayQueue sorted, const float3 *hitPositions,
                  const float3 *hitNormals, const int *hitMaterialIndices, int count);

GLOBAL
void reorderRays(const int *permutation, RayQueue current, RayQueue sorted, int count);

__global__ void computeMaterialRanges(const int *keys, int activeCount, MaterialRanges *ranges);

__global__ void buildOctantKeys(const float3 *directions, int count, unsigned int *keys, int *values);