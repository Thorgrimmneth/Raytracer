#pragma once
#include <vector>
#include "aabb.cuh"
#include "../raytracingUtils/hitrecord.cuh"    
#include "../objects/sphere.cuh"
#include "../objects/plane.cuh"
#include "../materials/material.cuh"
#include "../objects/triangle_mesh.cuh"

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
    int nbNodes;
    int nbObjects;

    
    __host__
    static BVHScene buildBVHScene(std::vector<BaseObject>* primitives,std::vector<Sphere>* spheres,std::vector<Plane>* planes,std::vector<TriangleMesh>* meshes);

    __host__
    size_t getDeviceSize() const;
    __device__ 
    bool intersect(const Ray& ray,
                   float tMin,
                   float tMax,
                   HitRecord& hit) const;

    __device__ __forceinline__
    bool intersectAny(const Ray& ray,
                  float tMin,
                  float tMax,
                  const Material* materials) const
    {
        int stack[32];
        int stackPtr = 0;
        stack[stackPtr++] = 0;

        while(stackPtr > 0)
        {
            int nodeIndex = stack[--stackPtr];
            const BVHSceneNode& node = d_nodes[nodeIndex];

            if (!node.bbox.intersect(ray, tMin, tMax))
                continue;

            if (node.isLeaf())
            {
                for(uint32_t i = node.firstIdx;
                    i < node.firstIdx + node.objectCount;
                    ++i)
                {
                    BaseObject& prim = d_primitives[d_indices[i]];

                    switch(prim.type)
                    {
                        case ObjectType::SPHERE:
                            if(materials[d_spheres[prim.index].materialIndex].type() == MaterialType::TRANSPARENT) continue;
                            if (d_spheres[prim.index].intersectAny(ray, tMin, tMax))
                                return true;
                            break;

                        case ObjectType::PLANE:
                            if(materials[d_planes[prim.index].materialIndex].type() == MaterialType::TRANSPARENT) continue;
                            if(d_planes[prim.index].intersectAny(ray, tMin, tMax, materials))
                                return true;
                            break;

                        case ObjectType::TRIANGLE:
                            if(materials[d_meshes[prim.index].materialIndex].type() == MaterialType::TRANSPARENT) continue;
                            if(d_meshes[prim.index].intersectAny(ray, tMin, tMax, materials))
                                return true;
                            break;
                    }
                }
            }
            else
            {
                stack[stackPtr++] = node.getLeftIndex();
                stack[stackPtr++] = node.right;
            }
        }

        return false;
    }
};

