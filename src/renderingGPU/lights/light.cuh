#pragma once

#include "lightsample.cuh"
#include "../utils/cuda_defines.cuh"
#include "../utils/op.cuh"

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

struct CudaScene;

enum LightType
{
    POINT,
    CYLINDER,
    DIRECTIONNAL,
    QUAD,
    SUN,
    SPHERE_GEOM,
    PLANE_GEOM,
    MESH_GEOM
};

struct Light
{
    float3 position;
    float height;
    float3 color;
    float radius;
    float3 direction;
    float area = 0.f;
    float3 u;
    float power;
    float3 v;
    LightType type;
    float3 normal;
    int geomIndex;

    __device__ LightSample sampleSphereGeom(const float3&p_point, curandState *rng, const CudaScene& scene) const;
    __device__ LightSample samplePlaneGeom(const float3&p_point, curandState *rng, const CudaScene& scene) const;
    __device__ LightSample sampleMeshGeom(const float3&p_point, curandState *rng, const CudaScene& scene) const;
    __device__ LightSample sampleCylinder(const float3&p_point, curandState *rng) const;
    __device__ LightSample sampleDirectionnal(const float3&p_point) const;
    __device__ LightSample samplePoint(const float3&p_point) const;
    __device__ LightSample sampleCone(const float3& p_point, curandState *rng) const;
    __device__ LightSample sampleQuad(const float3&, curandState*) const;
    __device__ LightSample sample(const float3& p_point, curandState *rng, const CudaScene& scene) const;
    __device__ LightSample sample(const float3&, curandState*) const;
    __device__ LightSample sample(const float3&) const;
};