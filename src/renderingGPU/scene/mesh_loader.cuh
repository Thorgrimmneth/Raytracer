#pragma once

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "../utils/simplifiedDef.cuh"
#include "../utils/op.cuh"
#include "../utils/quaternion.cuh"
#include <vector>

#include "../objects/base_object.cuh"
#include "../objects/triangle_mesh.cuh"
#include "../objects/plane.cuh"
#include <iostream>

// Load mesh geometry (shared data, loaded once)
HOST MeshGeometry loadMeshGeometry(const std::string &p_path);

// Load mesh geometry with scaling (shared data, loaded once)
HOST MeshGeometry loadMeshGeometry(const std::string &p_path, const float3 scale);

// Create instance with geometry reference
HOST MeshInstance createMeshInstance(int geometryIndex, int materialIndex, float3 scale = make_float3(1.f),
                           Quaternion rotation = quaternionFromAxisAngle(make_float3(0.f, 1.f, 0.f), 0.f), float3 translation = make_float3(0.f, 0.f, 0.f));

HOST MeshGeometry PlaneToMesh(const Plane &plane, float size = 20000.f);