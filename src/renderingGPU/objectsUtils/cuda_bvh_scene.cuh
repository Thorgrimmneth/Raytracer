#pragma once
#include <vector>
#include "cuda_aabb.cuh"
#include "../raytracingUtils/cuda_hitrecord.cuh"    
#include "../objects/cuda_sphere.cuh"
#include "../objects/cuda_plane.cuh"
#include "../materials/cuda_material.cuh"
#include "../objects/cuda_triangle_mesh.cuh"

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
                if(!(materials[d_spheres[i].materialIndex].type == MaterialType::TRANSPARENT)){
                    if(d_spheres[i].intersectAny(ray, tMin, tMax)) return true;
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

