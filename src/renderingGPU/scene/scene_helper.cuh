#pragma once

#include "../objects/sphere.cuh"
#include "../objects/implicit_sphere.cuh"
#include "../objects/plane.cuh"
#include "../objects/triangle_mesh.cuh"
#include "../materials/material.cuh"
#include "../lights/light.cuh"
#include "../objects/base_object.cuh"

struct CudaSceneHelper
{
    std::vector<Sphere> spheresGPU = std::vector<Sphere>();
    std::vector<ImplicitSphere> implicitSpheresGPU = std::vector<ImplicitSphere>();
    std::vector<Plane> planesGPU = std::vector<Plane>();
    std::vector<TriangleMesh> triangleMeshesGPU = std::vector<TriangleMesh>();
    std::vector<float3> verticesGPU = std::vector<float3>();
    std::vector<BaseObject> primitivesGPU = std::vector<BaseObject>();
    std::vector<Material> materialsGPU = std::vector<Material>();
    std::vector<Light> lightsGPU = std::vector<Light>();

    std::vector<Material> mirrorList = std::vector<Material>();
    std::vector<Material> transparentList = std::vector<Material>();
    std::vector<Material> emissiveList = std::vector<Material>();
    std::vector<Material> lambertList = std::vector<Material>();
    std::vector<Material> plasticList = std::vector<Material>();
    std::vector<Material> metalList = std::vector<Material>();

    std::vector<int> planeType = std::vector<int>();
    std::vector<int> sphereType = std::vector<int>();
    std::vector<int> triangleMeshType = std::vector<int>();
};