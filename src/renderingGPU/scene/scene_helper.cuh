#pragma once

#include "../utils/op.cuh"
#include "../objects/sphere.cuh"
#include "../objects/plane.cuh"
#include "../objects/triangle_mesh.cuh"
#include "../materials/material.cuh"
#include "../lights/light.cuh"
#include "../objects/sdf.cuh"

struct CudaSceneHelper
{
    std::vector<SDF> sdfsGPU = std::vector<SDF>();
    std::vector<Sphere> spheresGPU = std::vector<Sphere>();
    std::vector<Plane> planesGPU = std::vector<Plane>();
    std::vector<MeshGeometry> meshGeometriesGPU = std::vector<MeshGeometry>();  // Shared geometry data
    std::vector<MeshInstance> meshInstancesGPU = std::vector<MeshInstance>();    // Per-instance data
    std::vector<float3> verticesGPU = std::vector<float3>();
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