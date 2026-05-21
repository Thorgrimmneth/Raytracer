#pragma once

#include "../objects/triangle_mesh_geometry.cuh"
#include "../raytracingUtils/ray.cuh"
#include "aabb.cuh"
#include "bvh.cuh"
#include <algorithm>
#include <limits>
#include <vector>

struct TriangleRef
{
    int triIndex;
    AABB bbox;
    float3 centroid;
};

struct BVHBuildConfig
{
    int maxLeafSize = 4;
    int maxDepth = 64;

    // SBVH
    bool enableSpatialSplits = true;
    int binCount = 16;
    float traversalCost = 1.0f;
    float intersectionCost = 1.0f;

    // Pour éviter explosion mémoire par duplication
    float maxDuplicationRatio = 1.3f;
};

struct Bin
{
    AABB bbox;
    int count = 0;
};

enum SplitType
{
    SPLIT_NONE,
    SPLIT_OBJECT,
    SPLIT_SPATIAL
};

struct SplitCandidate
{
    SplitType type = SPLIT_NONE;
    int axis = 0;
    float pos = 0.f;
    float cost = std::numeric_limits<float>::infinity();
};

inline float getAxis(const float3 &v, int axis) { return axis == 0 ? v.x : axis == 1 ? v.y : v.z; }

inline float getAxisMin(const AABB &b, int axis) { return axis == 0 ? b.min.x : axis == 1 ? b.min.y : b.min.z; }

inline float getAxisMax(const AABB &b, int axis) { return axis == 0 ? b.max.x : axis == 1 ? b.max.y : b.max.z; }

inline void setAxisMin(AABB &b, int axis, float v)
{
    if (axis == 0)
        b.min.x = v;
    else if (axis == 1)
        b.min.y = v;
    else
        b.min.z = v;
}

inline void setAxisMax(AABB &b, int axis, float v)
{
    if (axis == 0)
        b.max.x = v;
    else if (axis == 1)
        b.max.y = v;
    else
        b.max.z = v;
}

SplitCandidate findBestSpatialSplit(const std::vector<TriangleRef> &refs, int start, int end, const AABB &parentBox,
                                    const BVHBuildConfig &config);

SplitCandidate findBestObjectSplit(std::vector<TriangleRef> &refs, int start, int end, const AABB &parentBox,
                                   const BVHBuildConfig &config);

int applyObjectSplit(std::vector<TriangleRef> &refs, int start, int end, const SplitCandidate &split);

int applySpatialSplit(std::vector<TriangleRef> &refs, int start, int end, const SplitCandidate &split);

int buildSBVHRecursive(std::vector<BVH> &nodes, std::vector<int> &finalRefs, const std::vector<TriangleRef> &refs,
                       const BVHBuildConfig &config, int depth);

HOST 
BVH *buildSBVH(TriangleMeshGeometry *triangles, int triangleCount, float3 *vertices, float3 *normals,
                        float2 *uvs, int &outNodeCount, int *&outTriangleRefIndices, int &outRefCount);