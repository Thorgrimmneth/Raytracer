#pragma once
#include <vector>
#include "aabb.cuh"
#include "../raytracingUtils/hitrecord.cuh"    
#include "../objects/sphere.cuh"
#include "../objects/plane.cuh"
#include "../materials/material.cuh"
#include "../objects/triangle_mesh.cuh"
#include "../objects/implicitSphere.cuh"
#include "../objects/base_object.cuh"

struct Current{
    uint32_t index;
    float distance;
};

struct BuildTask {
    uint32_t nodeIndex;
    uint32_t first;
    uint32_t last;
    uint8_t depth;
};

struct BVHSceneNode{
    AABB bbox;
    uint32_t left;          // 4 bytes (bit 31 = leaf flag, bits 0-30 = left index)
    uint32_t right;
    uint32_t firstIdx;
    uint32_t objectCount;
    // uint32_t version
    __device__ __forceinline__
    bool isLeaf() const { return (left & 0x80000000u) != 0; }
    
    __device__ __forceinline__
    uint32_t getLeftIndex() const { return left & 0x7FFFFFFFu; }

    // uint16_t version
    /*__device__ __forceinline__
    bool isLeaf() const { return (left & 0x8000u) != 0; }
    
    __device__ __forceinline__
    uint32_t getLeftIndex() const { return left & 0x7FFFu; }*/
};

struct BVHScene {

    BVHSceneNode* d_nodes;
    int* d_indices;

    BaseObject* d_primitives;
    Sphere* d_spheres;
    Plane* d_planes;
    TriangleMesh* d_meshes;
    ImplicitSphere* d_implicitSpheres;
    int nbNodes;
    int nbObjects;

    
    __host__
    static BVHScene buildBVHScene(std::vector<BaseObject>* primitives,std::vector<Sphere>* spheres,std::vector<TriangleMesh>* meshes, std::vector<ImplicitSphere>* implicitSpheres);

    __host__
    size_t getDeviceSize() const;

    __device__ __noinline__
    bool intersect(
    const Ray& ray,
    const float tMin,
    const float tMaxInit,
    HitRecord& hit) const
{
    constexpr int STACK_SIZE = 64;

    Current stack[STACK_SIZE];
    int stackPtr = 0;

    float tMax = tMaxInit;
    bool hitSomething = false;

#ifdef DEBUG_BVH
    if (d_nodes == nullptr ||
        d_primitives == nullptr ||
        d_indices == nullptr)
    {
        return false;
    }

    if (nbNodes <= 0 || nbObjects <= 0)
        return false;
#endif

    float rootTNear;
    float rootTFar;

    if (!d_nodes[0].bbox.intersect(ray, tMin, tMax, rootTNear, rootTFar))
        return false;

    stack[stackPtr++] = { 0, rootTNear };

    while (stackPtr > 0)
    {
        const Current current = stack[--stackPtr];

        if (current.distance > tMax)
            continue;

#ifdef DEBUG_BVH
        if ((unsigned)current.index >= (unsigned)nbNodes)
            continue;
#endif

        const BVHSceneNode& node = d_nodes[current.index];

        if (node.isLeaf())
        {
            const uint32_t first = node.firstIdx;
            const uint32_t count = node.objectCount;

#ifdef DEBUG_BVH
            if (first + count > nbObjects)
                continue;
#endif

            for (uint32_t i = first; i < first + count; ++i)
            {
                const int primArrayIndex = d_indices[i];

#ifdef DEBUG_BVH
                if ((unsigned)primArrayIndex >= (unsigned)nbObjects)
                    continue;
#endif

                const BaseObject& prim = d_primitives[primArrayIndex];
                const int objectIndex = prim.getIndex();

                switch (prim.getType())
                {
                    case ObjectType::SPHERE:
                    {
#ifdef DEBUG_BVH
                        if (d_spheres == nullptr)
                            break;
#endif
                        if (d_spheres[objectIndex].intersect(ray, tMin, tMax, hit))
                        {
                            tMax = hit.distance;
                            hitSomething = true;
                            hit.objectType = HIT_SPHERE;
                            hit.objectIndex = objectIndex;
                        }
                        break;
                    }

                    case ObjectType::TRIANGLE:
                    {
#ifdef DEBUG_BVH
                        if (d_meshes == nullptr)
                            break;
#endif
                        if (d_meshes[objectIndex].intersect(ray, tMin, tMax, hit))
                        {
                            tMax = hit.distance;
                            hitSomething = true;
                            hit.objectType = HIT_TRIANGLE_MESH;
                            hit.objectIndex = objectIndex;
                        }
                        break;
                    }

                    case ObjectType::IMPLICIT_SPHERE:
                    {
#ifdef DEBUG_BVH
                        if (d_implicitSpheres == nullptr)
                            break;
#endif
                        if (d_implicitSpheres[objectIndex].intersect(ray, tMin, tMax, hit))
                        {
                            tMax = hit.distance;
                            hitSomething = true;
                            hit.objectType = HIT_SPHERE_IMPLICIT;
                            hit.objectIndex = objectIndex;
                        }
                        break;
                    }

                    default:
                        break;
                }
            }
        }
        else
        {
            const uint32_t leftIdx = node.getLeftIndex();
            const uint32_t rightIdx = node.right;

#ifdef DEBUG_BVH
            if (leftIdx >= (uint32_t)nbNodes ||
                rightIdx >= (uint32_t)nbNodes)
            {
                continue;
            }
#endif

            float leftTNear;
            float leftTFar;
            float rightTNear;
            float rightTFar;

            const bool hitLeft = d_nodes[leftIdx].bbox.intersect(
                ray,
                tMin,
                tMax,
                leftTNear,
                leftTFar
            );

            const bool hitRight = d_nodes[rightIdx].bbox.intersect(
                ray,
                tMin,
                tMax,
                rightTNear,
                rightTFar
            );

            if (hitLeft && hitRight)
            {
                if (stackPtr + 2 > STACK_SIZE)
                    return hitSomething;

                // Stack LIFO : push le plus loin d'abord.
                if (leftTNear < rightTNear)
                {
                    stack[stackPtr++] = { rightIdx, rightTNear };
                    stack[stackPtr++] = { leftIdx, leftTNear };
                }
                else
                {
                    stack[stackPtr++] = { leftIdx, leftTNear };
                    stack[stackPtr++] = { rightIdx, rightTNear };
                }
            }
            else if (hitLeft)
            {
                if (stackPtr + 1 > STACK_SIZE)
                    return hitSomething;

                stack[stackPtr++] = { leftIdx, leftTNear };
            }
            else if (hitRight)
            {
                if (stackPtr + 1 > STACK_SIZE)
                    return hitSomething;

                stack[stackPtr++] = { rightIdx, rightTNear };
            }
        }
    }

    return hitSomething;
}

    __device__
bool intersectAny(
    const Ray& ray,
    float tMin,
    float tMax,
    const Material* materials) const
{
    constexpr int STACK_SIZE = 64;

    Current stack[STACK_SIZE];
    int stackPtr = 0;

#ifdef DEBUG_BVH
    if (d_nodes == nullptr ||
        d_primitives == nullptr ||
        d_indices == nullptr)
    {
        return false;
    }

    if (nbNodes <= 0 || nbObjects <= 0)
        return false;
#endif

    float rootTNear;
    float rootTFar;

    if (!d_nodes[0].bbox.intersect(ray, tMin, tMax, rootTNear, rootTFar))
        return false;

    stack[stackPtr++] = { 0, rootTNear };

    while (stackPtr > 0)
    {
        const Current current = stack[--stackPtr];

        if (current.distance > tMax)
            continue;

#ifdef DEBUG_BVH
        if ((unsigned)current.index >= (unsigned)nbNodes)
            continue;
#endif

        const BVHSceneNode& node = d_nodes[current.index];

        if (node.isLeaf())
        {
            const uint32_t first = node.firstIdx;
            const uint32_t count = node.objectCount;

#ifdef DEBUG_BVH
            if (first + count > nbObjects)
                continue;
#endif

            for (uint32_t i = first; i < first + count; ++i)
            {
                const int primArrayIndex = d_indices[i];

#ifdef DEBUG_BVH
                if ((unsigned)primArrayIndex >= (unsigned)nbObjects)
                    continue;
#endif

                const BaseObject& prim = d_primitives[primArrayIndex];
                const int objectIndex = prim.getIndex();

                switch (prim.getType())
                {
                    case ObjectType::SPHERE:
                    {
#ifdef DEBUG_BVH
                        if (d_spheres == nullptr)
                            break;
#endif
                        const int matIdx = d_spheres[objectIndex].materialIndex;

                        if (materials != nullptr &&
                            materials[matIdx].type() == MaterialType::TRANSPARENT)
                        {
                            break;
                        }

                        if (d_spheres[objectIndex].intersectAny(ray, tMin, tMax))
                            return true;

                        break;
                    }

                    case ObjectType::TRIANGLE:
                    {
#ifdef DEBUG_BVH
                        if (d_meshes == nullptr)
                            break;
#endif
                        const int matIdx = d_meshes[objectIndex].materialIndex;

                        if (materials != nullptr &&
                            materials[matIdx].type() == MaterialType::TRANSPARENT)
                        {
                            break;
                        }

                        if (d_meshes[objectIndex].intersectAny(ray, tMin, tMax, materials))
                            return true;

                        break;
                    }

                    case ObjectType::IMPLICIT_SPHERE:
                    {
#ifdef DEBUG_BVH
                        if (d_implicitSpheres == nullptr)
                            break;
#endif
                        const int matIdx = d_implicitSpheres[objectIndex].materialIndex;

                        if (materials != nullptr &&
                            materials[matIdx].type() == MaterialType::TRANSPARENT)
                        {
                            break;
                        }

                        if (d_implicitSpheres[objectIndex].intersectAny(ray, tMin, tMax))
                            return true;

                        break;
                    }

                    default:
                        break;
                }
            }
        }
        else
        {
            const uint32_t leftIdx = node.getLeftIndex();
            const uint32_t rightIdx = node.right;

#ifdef DEBUG_BVH
            if (leftIdx >= (uint32_t)nbNodes ||
                rightIdx >= (uint32_t)nbNodes)
            {
                continue;
            }
#endif

            float leftTNear;
            float leftTFar;
            float rightTNear;
            float rightTFar;

            const bool hitLeft = d_nodes[leftIdx].bbox.intersect(
                ray,
                tMin,
                tMax,
                leftTNear,
                leftTFar
            );

            const bool hitRight = d_nodes[rightIdx].bbox.intersect(
                ray,
                tMin,
                tMax,
                rightTNear,
                rightTFar
            );

            if (hitLeft && hitRight)
            {
                if (stackPtr + 2 > STACK_SIZE)
                    return false;

                if (leftTNear < rightTNear)
                {
                    stack[stackPtr++] = { rightIdx, rightTNear };
                    stack[stackPtr++] = { leftIdx, leftTNear };
                }
                else
                {
                    stack[stackPtr++] = { leftIdx, leftTNear };
                    stack[stackPtr++] = { rightIdx, rightTNear };
                }
            }
            else if (hitLeft)
            {
                if (stackPtr + 1 > STACK_SIZE)
                    return false;

                stack[stackPtr++] = { leftIdx, leftTNear };
            }
            else if (hitRight)
            {
                if (stackPtr + 1 > STACK_SIZE)
                    return false;

                stack[stackPtr++] = { rightIdx, rightTNear };
            }
        }
    }

    return false;
}

};

