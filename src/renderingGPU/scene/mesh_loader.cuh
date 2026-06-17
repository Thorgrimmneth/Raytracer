#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <vector>
#include "../utils/quaternion.cuh"
#include "../utils/op.cuh"
#include "../utils/macro.cuh"

#include "../objects/base_object.cuh"
#include "../objects/triangle_mesh.cuh"


HOST
TriangleMesh loadTriangleMesh(const std::string &p_path, int materialIndex, int index, 
                                        float3 scale, Quaternion rotation, float3 translation);