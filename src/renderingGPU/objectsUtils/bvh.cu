#include "bvh.cuh"
#include "../objects/triangle_mesh.cuh"

#include <iostream>
#include <cstdint>
#include <cstring>
__host__
BVH* buildBVH(
    TriangleMeshGeometry* triangles,
    int triangleCount,
    float3* vertices,
    float3* normals,
    float2* uvs,
    int& outNodeCount)
{
    std::vector<BVH> bvhNodes;
    bvhNodes.push_back(BVH());
    std::vector<int> triangleIndices(triangleCount);

    for (int i = 0; i < triangleCount; ++i)
        triangleIndices[i] = i;

    struct BuildTask {
        int nodeIdx;
        int triStart;
        int triEnd;
    };

    std::vector<BuildTask> taskStack;
    taskStack.push_back({0, 0, triangleCount});

    while (!taskStack.empty()) {
        BuildTask task = taskStack.back();
        taskStack.pop_back();

        int nodeIdx = task.nodeIdx;
        int triStart = task.triStart;
        int triEnd = task.triEnd;

        while ((int)bvhNodes.size() <= nodeIdx)
            bvhNodes.push_back(BVH());

        BVH& node = bvhNodes[nodeIdx];

        node.firstTriangleIndex = triStart;
        node.lastTriangleIndex = triEnd - 1;

        node.bbox = AABB();

        for (int i = triStart; i < triEnd; ++i) {
            int triIdx = triangleIndices[i];
            const TriangleMeshGeometry& tri = triangles[triIdx];

            node.bbox.extend(vertices[tri.i0]);
            node.bbox.extend(vertices[tri.i1]);
            node.bbox.extend(vertices[tri.i2]);
        }

        if (triEnd - triStart <= 4) {
            node.left = -1;
            node.right = -1;
            continue;
        }

        float3 extent = make_float3(
            node.bbox.max.x - node.bbox.min.x,
            node.bbox.max.y - node.bbox.min.y,
            node.bbox.max.z - node.bbox.min.z
        );

        int splitAxis = 0;
        if (extent.y > extent.x) splitAxis = 1;
        if (extent.z > ((splitAxis == 0) ? extent.x : extent.y)) splitAxis = 2;

        float bboxMinAxis =
            splitAxis == 0 ? node.bbox.min.x :
            splitAxis == 1 ? node.bbox.min.y :
                             node.bbox.min.z;

        float extentAxis =
            splitAxis == 0 ? extent.x :
            splitAxis == 1 ? extent.y :
                             extent.z;

        float splitPos = bboxMinAxis + extentAxis * 0.5f;

        int leftEnd = triStart;

        for (int i = triStart; i < triEnd; ++i) {
            int triIdx = triangleIndices[i];
            const TriangleMeshGeometry& tri = triangles[triIdx];

            float3 v0 = vertices[tri.i0];
            float3 v1 = vertices[tri.i1];
            float3 v2 = vertices[tri.i2];

            float c0 =
                splitAxis == 0 ? v0.x :
                splitAxis == 1 ? v0.y :
                                 v0.z;

            float c1 =
                splitAxis == 0 ? v1.x :
                splitAxis == 1 ? v1.y :
                                 v1.z;

            float c2 =
                splitAxis == 0 ? v2.x :
                splitAxis == 1 ? v2.y :
                                 v2.z;

            float centroid = (c0 + c1 + c2) / 3.f;

            if (centroid < splitPos) {
                std::swap(triangleIndices[i], triangleIndices[leftEnd]);
                ++leftEnd;
            }
        }

        if (leftEnd == triStart || leftEnd == triEnd)
            leftEnd = triStart + (triEnd - triStart) / 2;

        int leftChild = (int)bvhNodes.size();
        bvhNodes.push_back(BVH());

        int rightChild = (int)bvhNodes.size();
        bvhNodes.push_back(BVH());

        bvhNodes[nodeIdx].left = leftChild;
        bvhNodes[nodeIdx].right = rightChild;

        taskStack.push_back({rightChild, leftEnd, triEnd});
        taskStack.push_back({leftChild, triStart, leftEnd});
    }

    // Important: reorder triangles according to BVH leaf layout
    std::vector<TriangleMeshGeometry> orderedTriangles(triangleCount);

    for (int i = 0; i < triangleCount; ++i) {
        orderedTriangles[i] = triangles[triangleIndices[i]];
    }

    std::memcpy(
        triangles,
        orderedTriangles.data(),
        triangleCount * sizeof(TriangleMeshGeometry)
    );

    BVH* d_bvhNodes = nullptr;
    cudaMalloc(&d_bvhNodes, bvhNodes.size() * sizeof(BVH));
    cudaMemcpy(
        d_bvhNodes,
        bvhNodes.data(),
        bvhNodes.size() * sizeof(BVH),
        cudaMemcpyHostToDevice
    );

    outNodeCount = (int)bvhNodes.size();
    return d_bvhNodes;
}