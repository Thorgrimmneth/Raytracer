#include "sortQueues.cuh"

GLOBAL
void classifyPairs(Material *materials, int activeCount, int *keys, int *values, const int *hitMask,
                   const int *hitMaterialIndices)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    if (!hitMask[qid])
    {
        keys[qid] = (int)MISS;
        values[qid] = qid;
        return;
    }

    int materialIndex = hitMaterialIndices[qid];
    const Material &mtl = materials[materialIndex];

    keys[qid] = (int)mtl.baseColor.w;
    values[qid] = qid;
}

GLOBAL
void reorderPaths(const int *permutation, const float3 *origins, const float3 *directions, const float3 *throughput,
                  const float3 *hitPositions, const float3 *hitNormals, const int *hitMaterialIndices,
                  const int *pixelIndices, const bool *lastBounceWasDelta, const float *lastBsdfPdf,
                  const bool *isInside, const RNG *rng, float3 *sortedOrigins, float3 *sortedDirections,
                  float3 *sortedThroughput, float3 *sortedHitPositions, float3 *sortedHitNormals,
                  int *sortedHitMaterialIndices, int *sortedPixelIndices, bool *sortedLastBounceWasDelta,
                  float *sortedLastBsdfPdf, bool *sortedIsInside, RNG *sortedRNG, int count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= count)
        return;

    int src = permutation[tid];

    sortedOrigins[tid] = origins[src];
    sortedDirections[tid] = directions[src];
    sortedThroughput[tid] = throughput[src];

    sortedHitPositions[tid] = hitPositions[src];
    sortedHitNormals[tid] = hitNormals[src];

    sortedHitMaterialIndices[tid] = hitMaterialIndices[src];
    sortedPixelIndices[tid] = pixelIndices[src];

    sortedLastBounceWasDelta[tid] = lastBounceWasDelta[src];
    sortedLastBsdfPdf[tid] = lastBsdfPdf[src];
    sortedIsInside[tid] = isInside[src];
    sortedRNG[tid] = rng[src];
}

__global__ void computeMaterialRanges(const int *keys, int activeCount, MaterialRanges *ranges)
{
    int mat = threadIdx.x;

    if (mat >= 7)
        return;

    // lower_bound
    int left = 0;
    int right = activeCount;

    while (left < right)
    {
        int mid = (left + right) >> 1;

        if (keys[mid] < mat)
            left = mid + 1;
        else
            right = mid;
    }

    int begin = left;

    // upper_bound
    left = begin;
    right = activeCount;

    while (left < right)
    {
        int mid = (left + right) >> 1;

        if (keys[mid] <= mat)
            left = mid + 1;
        else
            right = mid;
    }

    int end = left;

    ranges->offset[mat] = begin;
    ranges->count[mat] = end - begin;
}

__global__ void buildOctantKeys(const float3 *directions, int count, unsigned int *keys, int *values)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= count)
        return;

    float3 d = directions[tid];

    unsigned int octant = (d.x > 0.f) | ((d.y > 0.f) << 1) | ((d.z > 0.f) << 2);

    keys[tid] = octant;
    values[tid] = tid;
}