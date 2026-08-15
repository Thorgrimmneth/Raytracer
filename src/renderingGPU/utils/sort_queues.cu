#include "sort_queues.cuh"

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
void reorderPaths(const int *__restrict__ permutation, RayQueue current, SortedRayQueue sorted,
                  const float3 *__restrict__ hitPositions, const float3 *__restrict__ hitNormals,
                  const int *__restrict__ hitMaterialIndices, const float *__restrict__ distances,
                  const int *__restrict__ objectIndices, const int *__restrict__ types, int count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= count)
        return;

    const int src = permutation[tid];

    sorted.origins[tid] = current.origins[src];
    sorted.directions[tid] = current.directions[src];
    sorted.throughputs[tid] = current.throughputs[src];

    sorted.hitPositions[tid] = hitPositions[src];
    sorted.hitNormals[tid] = hitNormals[src];

    sorted.hitMaterialIndices[tid] = hitMaterialIndices[src];

    sorted.hitObjectIndices[tid] = objectIndices[src];

    sorted.hitDistances[tid] = distances[src];

    sorted.hitTypes[tid] = types[src];

    sorted.pixelIndices[tid] = current.pixelIndices[src];

    sorted.lastBounceWasDelta[tid] = current.lastBounceWasDelta[src];

    sorted.lastBsdfPdf[tid] = current.lastBsdfPdf[src];

    sorted.isInside[tid] = current.isInside[src];

    sorted.rng[tid] = current.rng[src];
}

GLOBAL
void reorderRays(const int *permutation, RayQueue current, RayQueue next, int count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= count)
        return;

    int src = permutation[tid];

    next.origins[tid] = current.origins[src];
    next.directions[tid] = current.directions[src];
    next.throughputs[tid] = current.throughputs[src];

    next.pixelIndices[tid] = current.pixelIndices[src];

    next.lastBounceWasDelta[tid] = current.lastBounceWasDelta[src];
    next.lastBsdfPdf[tid] = current.lastBsdfPdf[src];
    next.isInside[tid] = current.isInside[src];
    next.rng[tid] = current.rng[src];
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