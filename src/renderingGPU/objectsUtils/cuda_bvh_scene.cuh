#pragma once
#include <vector>
#include "cuda_aabb.cuh"
#include "../raytracingUtils/cuda_hitrecord.cuh"    
#include "../objects/cuda_sphere.cuh"
#include "../materials/cuda_material.cuh"

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
    Sphere* d_objects;

    int nbNodes;
    int nbObjects;

    
    __host__
    static BVHScene buildBVHScene(std::vector<Sphere>* objects);

    __device__
    bool intersect(const Ray& ray,
                   float tMin,
                   float tMax,
                   HitRecord& hit) const;

    __device__
    bool intersectAny(const Ray& ray,
                      float tMin,
                      float tMax, 
                      const Material* materials) const;
};

