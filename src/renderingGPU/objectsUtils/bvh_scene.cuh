#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "../utils/simplified_def.cuh"

#include "aabb.cuh"

#include "../objects/base_object.cuh"
#include "../objects/implicit_sphere.cuh"
#include "../objects/plane.cuh"
#include "../objects/sphere.cuh"
#include "../objects/triangle_mesh.cuh"

#include "../materials/material.cuh"

#include "../../devicePrograms/launch_radiance_params.cuh"

struct Current
{
    uint32_t index;
    float distance;
};

struct BuildTask
{
    uint32_t nodeIndex;
    uint32_t first;
    uint32_t last;
    uint8_t depth;
};

struct BVHSceneNode
{
    AABB bbox;
    uint32_t left; // 4 bytes (bit 31 = leaf flag, bits 0-30 = left index)
    uint32_t right;
    uint32_t firstIdx;
    uint32_t objectCount;
    // uint32_t version
    D_FORCEINLINE bool isLeaf() const { return (left & 0x80000000u) != 0; }

    D_FORCEINLINE uint32_t getLeftIndex() const { return left & 0x7FFFFFFFu; }

    // uint16_t version
    /*D_FORCEINLINE
    bool isLeaf() const { return (left & 0x8000u) != 0; }

    D_FORCEINLINE
    uint32_t getLeftIndex() const { return left & 0x7FFFu; }*/
};

struct BVHScene
{

    BVHSceneNode *d_nodes;
    int *d_indices;

    BaseObject *d_primitives;

    Sphere *d_spheres;
    ImplicitSphere *d_implicitSpheres;
    Plane *d_planes;
    
    int nbNodes;
    int nbObjects;

    HOST 
    static BVHScene buildBVHScene(std::vector<BaseObject> *primitives, std::vector<Sphere> *spheres,
                                           
                                           std::vector<ImplicitSphere> *implicitSpheres);

    HOST 
    size_t getDeviceSize() const;

    /*D_FORCEINLINE 
    bool intersect(const float3& origin, const float3 &direction, const float tMin, const float tMaxInit, OptixHit &hit) const
    {
        constexpr int STACK_SIZE = 32;

        Current stack[STACK_SIZE];
        int stackPtr = 0;

        float tMax = tMaxInit;
        bool hitSomething = false;

        float rootTNear;

        if (!d_nodes[0].bbox.intersectCheck(origin, direction, tMin, tMax, rootTNear))
            return false;

        stack[stackPtr++] = {0, rootTNear};

        while (stackPtr > 0)
        {
            const Current current = stack[--stackPtr];

            if (current.distance > tMax)
                continue;

            const BVHSceneNode &node = d_nodes[current.index];

            if (node.isLeaf())
            {
                const uint32_t first = node.firstIdx;
                const uint32_t count = node.objectCount;

                for (uint32_t i = first; i < first + count; ++i)
                {
                    const int primArrayIndex = d_indices[i];

                    const BaseObject &prim = d_primitives[primArrayIndex];
                    const int objectIndex = prim.getIndex();

                    switch (prim.getType())
                    {
                    case object_type::SPHERE: {

                        if (d_spheres[objectIndex].intersect(origin, direction, tMin, tMax, hit))
                        {
                            tMax = hit.t;
                            hitSomething = true;
                            hit.object_type = HIT_SPHERE;
                            hit.objectIndex = objectIndex;
                        }
                        break;
                    }

                    case object_type::IMPLICIT_SPHERE: {

                        if (d_implicitSpheres[objectIndex].intersect(origin, direction, tMin, tMax, hit))
                        {
                            tMax = hit.t;
                            hitSomething = true;
                            hit.object_type = HIT_SPHERE_IMPLICIT;
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

                float leftTNear;
                float rightTNear;

                const bool hitLeft = d_nodes[leftIdx].bbox.intersectCheck(origin, direction, tMin, tMax, leftTNear);

                const bool hitRight = d_nodes[rightIdx].bbox.intersectCheck(origin, direction, tMin, tMax, rightTNear);

                if (hitLeft && hitRight)
                {
                    if (stackPtr + 2 > STACK_SIZE)
                        return hitSomething;

                    // Stack LIFO : push le plus loin d'abord.
                    if (leftTNear < rightTNear)
                    {
                        stack[stackPtr++] = {rightIdx, rightTNear};
                        stack[stackPtr++] = {leftIdx, leftTNear};
                    }
                    else
                    {
                        stack[stackPtr++] = {leftIdx, leftTNear};
                        stack[stackPtr++] = {rightIdx, rightTNear};
                    }
                }
                else if (hitLeft)
                {
                    if (stackPtr + 1 > STACK_SIZE)
                        return hitSomething;

                    stack[stackPtr++] = {leftIdx, leftTNear};
                }
                else if (hitRight)
                {
                    if (stackPtr + 1 > STACK_SIZE)
                        return hitSomething;

                    stack[stackPtr++] = {rightIdx, rightTNear};
                }
            }
        }

        return hitSomething;
    }

    D_FORCEINLINE 
    bool intersectAny(const float3 &origin, const float3 &direction, float tMin, float tMax, const Material *materials) const
    {
        constexpr int STACK_SIZE = 32;

        Current stack[STACK_SIZE];
        int stackPtr = 0;

        float rootTNear;

        if (!d_nodes[0].bbox.intersectCheck(origin, direction, tMin, tMax, rootTNear))
            return false;

        stack[stackPtr++] = {0, rootTNear};

        while (stackPtr > 0)
        {
            const Current current = stack[--stackPtr];

            if (current.distance > tMax)
                continue;

            const BVHSceneNode &node = d_nodes[current.index];

            if (node.isLeaf())
            {
                const uint32_t first = node.firstIdx;
                const uint32_t count = node.objectCount;

                for (uint32_t i = first; i < first + count; ++i)
                {
                    const int primArrayIndex = d_indices[i];

                    const BaseObject &prim = d_primitives[primArrayIndex];
                    const int objectIndex = prim.getIndex();

                    switch (prim.getType())
                    {
                    case object_type::SPHERE: {

                        const int matIdx = d_spheres[objectIndex].getMaterialIndex();

                        if (materials[matIdx].type() == MaterialType::TRANSPARENT)
                        {
                            break;
                        }

                        if (d_spheres[objectIndex].intersectAny(origin, direction, tMin, tMax))
                            return true;

                        break;
                    }

                    case object_type::IMPLICIT_SPHERE: {

                        const int matIdx = d_implicitSpheres[objectIndex].getMaterialIndex();

                        if (materials[matIdx].type() == MaterialType::TRANSPARENT)
                        {
                            break;
                        }

                        if (d_implicitSpheres[objectIndex].intersectAny(origin, direction, tMin, tMax))
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

                float leftTNear;
                float rightTNear;

                const bool hitLeft = d_nodes[leftIdx].bbox.intersectCheck(origin, direction, tMin, tMax, leftTNear);

                const bool hitRight = d_nodes[rightIdx].bbox.intersectCheck(origin, direction, tMin, tMax, rightTNear);

                if (hitLeft && hitRight)
                {
                    if (stackPtr + 2 > STACK_SIZE)
                        return true;

                    if (leftTNear < rightTNear)
                    {
                        stack[stackPtr++] = {rightIdx, rightTNear};
                        stack[stackPtr++] = {leftIdx, leftTNear};
                    }
                    else
                    {
                        stack[stackPtr++] = {leftIdx, leftTNear};
                        stack[stackPtr++] = {rightIdx, rightTNear};
                    }
                }
                else if (hitLeft)
                {
                    if (stackPtr + 1 > STACK_SIZE)
                        return true;

                    stack[stackPtr++] = {leftIdx, leftTNear};
                }
                else if (hitRight)
                {
                    if (stackPtr + 1 > STACK_SIZE)
                        return true;

                    stack[stackPtr++] = {rightIdx, rightTNear};
                }
            }
        }

        return false;
    }*/
};
