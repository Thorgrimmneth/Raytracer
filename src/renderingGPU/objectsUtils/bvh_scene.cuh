#pragma once
#include <vector>
#include "aabb.cuh"
#include "../raytracingUtils/hitrecord.cuh"    
#include "../objects/sphere.cuh"
#include "../objects/plane.cuh"
#include "../materials/material.cuh"
#include "../objects/triangle_mesh.cuh"

struct Current{
    int index;
    float distance;
};

struct BuildTask {
    int nodeIndex;
    int first;
    int last;
    int depth;
};

struct BVHSceneNode{
    AABB bbox;
    int left = -1;
    int right = -1;
    int firstObjectIndex = -1;
    int lastObjectIndex = -1;

    __device__
    inline bool isLeaf() const { return ( left == -1); }
};

struct BVHScene {

    BVHSceneNode* d_nodes;
    BaseObject* d_primitives;
    Sphere* d_spheres;
    Plane* d_planes;
    TriangleMesh* d_meshes;
    int nbNodes;
    int nbObjects;

    
    __host__
    static BVHScene buildBVHScene(std::vector<BaseObject>* primitives,std::vector<Sphere>* spheres,std::vector<Plane>* planes,std::vector<TriangleMesh>* meshes);

    __device__ 
    bool intersect(const Ray& ray,
                   float tMin,
                   float tMax,
                   HitRecord& hit) const;

    __device__ __forceinline__
    bool intersectAny(const Ray& ray,
                  float tMin,
                  float tMax,
                  const Material* materials) const{
    int stack[32];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // root index

    while(stackPtr > 0)
    {
        int nodeIndex = stack[--stackPtr];
        const BVHSceneNode& node = d_nodes[nodeIndex];

        if (!node.bbox.intersect(ray, tMin, tMax))
            continue;

        if (node.isLeaf())
        {
            for(int i = node.firstObjectIndex;
                i < node.lastObjectIndex;
                ++i)
            {
                BaseObject& prim = d_primitives[i];
                switch(prim.type)
                {
                    case ObjectType::SPHERE:
                    if(materials[d_spheres[prim.index].materialIndex].type == MaterialType::TRANSPARENT) continue;
                        if (d_spheres[prim.index].intersectAny(ray, tMin, tMax))
                        {
                            return true;
                        }
                        break;
                    case ObjectType::PLANE:
                    if(materials[d_planes[prim.index].materialIndex].type == MaterialType::TRANSPARENT) continue;
                        if(d_planes[prim.index].intersectAny(ray, tMin, tMax, materials))
                        {
                            return true;
                        }
                        break;
                    case ObjectType::TRIANGLE:
                    if(materials[d_meshes[prim.index].materialIndex].type == MaterialType::TRANSPARENT) continue;
                        if(d_meshes[prim.index].intersectAny(ray, tMin, tMax, materials))
                        {
                            return true;
                        }
                        break;
                }
            }
        }
        else
        {
            stack[stackPtr++] = node.left;
            stack[stackPtr++] = node.right;
        }
    }

    return false;
    }
};

