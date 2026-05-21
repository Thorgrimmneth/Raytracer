#pragma once

#include "triangle_mesh_geometry.cuh"

#include "../utils/objects.cuh"

#include "../materials/material.cuh"
#include "../objectsUtils/bvh.cuh"

struct TriangleMesh
{
    TriangleMeshGeometry *triangles = nullptr;
    float3 *vertices = nullptr;
    float3 *normals = nullptr;
    float2 *uvs = nullptr;

    int triangleCount = 0;
    int vertexCount = 0;

    BVH *bvhNodes = nullptr;
    int bvhNodeCount = 0;

    int *triangleRefIndices = nullptr;
    int refCount = 0;

    int materialIndex = 0;

    float meshArea = 0.0f;
    float *triangleAreaCdf = nullptr;

    D_FORCEINLINE 
    bool intersect(const Ray &p_ray, const float p_tMin, const float p_tMax,
                                            HitRecord &p_hitRecord) const
    {
        constexpr int STACK_SIZE = 64;

        int stack[STACK_SIZE];
        int stackPtr = 0;

        stack[stackPtr++] = 0;

        float tClosest = p_tMax;
        bool hit = false;

        while (stackPtr > 0)
        {
            const int nodeIndex = stack[--stackPtr];

            const BVH &node = bvhNodes[nodeIndex];

            float nodeTNear;

            if (!node.bbox.intersectCheck(p_ray, p_tMin, tClosest, nodeTNear))
            {
                continue;
            }

            if (node.isLeaf())
            {
                const int first = node.firstRefIndex;
                const int count = node.refCount;

                for (int localIdx = 0; localIdx < count; ++localIdx)
                {
                    const int refIndex = first + localIdx;

                    const int triIndex = triangleRefIndices[refIndex];

                    const TriangleMeshGeometry &tri = triangles[triIndex];

                    float t;
                    float2 uv;

                    if (tri.intersect(p_ray, p_tMin, tClosest, t, uv))
                    {
                        tClosest = t;
                        hit = true;

                        p_hitRecord.point = p_ray.pointAtT(t);
                        p_hitRecord.normal = tri.computeSmoothNormal(uv, normals);
                        p_hitRecord.faceNormal(p_ray.direction);
                        p_hitRecord.distance = t;
                        p_hitRecord.materialIndex = materialIndex;
                    }
                }
            }
            else
            {
                const int left = node.left;
                const int right = node.right;

                float leftTNear;
                float rightTNear;

                const bool hitLeft = bvhNodes[left].bbox.intersectCheck(p_ray, p_tMin, tClosest, leftTNear);

                const bool hitRight = bvhNodes[right].bbox.intersectCheck(p_ray, p_tMin, tClosest, rightTNear);

                if (hitLeft && hitRight)
                {
                    if (stackPtr + 2 > STACK_SIZE)
                        return hit;

                    if (leftTNear < rightTNear)
                    {
                        stack[stackPtr++] = right;
                        stack[stackPtr++] = left;
                    }
                    else
                    {
                        stack[stackPtr++] = left;
                        stack[stackPtr++] = right;
                    }
                }
                else if (hitLeft)
                {
                    if (stackPtr + 1 > STACK_SIZE)
                        return hit;

                    stack[stackPtr++] = left;
                }
                else if (hitRight)
                {
                    if (stackPtr + 1 > STACK_SIZE)
                        return hit;

                    stack[stackPtr++] = right;
                }
            }
        }

        return hit;
    }

    D_FORCEINLINE 
    bool intersectAny(const Ray &p_ray, const float p_tMin, const float p_tMax,
                                                  const Material *materials) const
    {
        constexpr int STACK_SIZE = 64;

        int stack[STACK_SIZE];
        int stackPtr = 0;

        stack[stackPtr++] = 0;

        while (stackPtr > 0)
        {
            const int nodeIndex = stack[--stackPtr];

            const BVH &node = bvhNodes[nodeIndex];

            float nodeTNear;

            if (!node.bbox.intersectCheck(p_ray, p_tMin, p_tMax, nodeTNear))
            {
                continue;
            }

            if (node.isLeaf())
            {
                const int first = node.firstRefIndex;
                const int count = node.refCount;

                for (int localIdx = 0; localIdx < count; ++localIdx)
                {
                    const int refIndex = first + localIdx;

                    const int triIndex = triangleRefIndices[refIndex];

                    const TriangleMeshGeometry &tri = triangles[triIndex];

                    float t;
                    float2 uv;

                    if (tri.intersect(p_ray, p_tMin, p_tMax, t, uv))
                    {
                        return true;
                    }
                }
            }
            else
            {
                const int left = node.left;
                const int right = node.right;

                float leftTNear;
                float rightTNear;

                const bool hitLeft = bvhNodes[left].bbox.intersectCheck(p_ray, p_tMin, p_tMax, leftTNear);

                const bool hitRight = bvhNodes[right].bbox.intersectCheck(p_ray, p_tMin, p_tMax, rightTNear);

                if (hitLeft && hitRight)
                {
                    if (stackPtr + 2 > STACK_SIZE)
                        return false;

                    // Stack LIFO :
                    // on push le plus loin d'abord pour visiter le plus proche en premier.
                    if (leftTNear < rightTNear)
                    {
                        stack[stackPtr++] = right;
                        stack[stackPtr++] = left;
                    }
                    else
                    {
                        stack[stackPtr++] = left;
                        stack[stackPtr++] = right;
                    }
                }
                else if (hitLeft)
                {
                    if (stackPtr + 1 > STACK_SIZE)
                        return false;

                    stack[stackPtr++] = left;
                }
                else if (hitRight)
                {
                    if (stackPtr + 1 > STACK_SIZE)
                        return false;

                    stack[stackPtr++] = right;
                }
            }
        }

        return false;
    }
};