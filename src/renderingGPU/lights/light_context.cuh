#pragma once

#include "../objects/sdf.cuh"
#include "../objects/triangle_mesh.cuh"
#include "../materials/material.cuh"

struct LightContext
{
    const SDF* sdfs;
    const MeshInstance* meshInstances;
    const MeshGeometry* meshGeometries;
    const Material* materials;
};