#pragma once

#include "../objects/triangle_mesh.cuh"

#include <algorithm>

inline void computeTransform(const MeshInstance &mesh, OptixInstance &instance)
{
    std::copy_n(mesh.transform, 12, instance.transform);
}