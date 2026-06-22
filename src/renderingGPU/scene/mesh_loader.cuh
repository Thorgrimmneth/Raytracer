#pragma once

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "../utils/macro.cuh"
#include "../utils/op.cuh"
#include "../utils/quaternion.cuh"
#include <vector>

#include "../objects/base_object.cuh"
#include "../objects/triangle_mesh.cuh"
#include "../objects/plane.cuh"

// Load mesh geometry (shared data, loaded once)
HOST MeshGeometry loadMeshGeometry(const std::string &p_path);

// Load mesh geometry with scaling (shared data, loaded once)
HOST MeshGeometry loadMeshGeometry(const std::string &p_path, const float3 scale);

// Load mesh with transform (legacy, applies transform immediately)
HOST TriangleMesh loadTriangleMesh(const std::string &p_path, int materialIndex, int index, float3 scale,
                                   Quaternion rotation, float3 translation);

// Create instance with geometry reference
HOST MeshInstance createMeshInstance(int geometryIndex, int materialIndex, float3 scale, 
                                    Quaternion rotation, float3 translation);

HOST TriangleMesh PlaneToMesh(const Plane &plane, float size = 20000.f);