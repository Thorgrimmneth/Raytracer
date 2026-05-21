#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "objectsUtils/bvh.cuh"
#include "objectsUtils/sbvh.cuh"
#include "utils/quaternion.cuh"
#include "objects/triangle_mesh.cuh"
#include "objects/triangle_mesh_geometry.cuh"
#include "utils/op.cuh"
#include "utils/macro.cuh"
#include "objects/base_object.cuh"

struct MeshAndPrimitive
{
    TriangleMesh mesh;
    BaseObject prim;

    MeshAndPrimitive(TriangleMesh p_mesh, float3 min, float3 max, ObjectType type, int index)
        : prim(BaseObject(min, max, type, index)), mesh(p_mesh)
    {
    }
};

HOST
MeshAndPrimitive loadTriangleMesh(const std::string &p_path, int materialIndex, int index, 
                                        float3 scale, Quaternion rotation, float3 translation);