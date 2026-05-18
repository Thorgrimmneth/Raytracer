#pragma once

#include "lightsample.cuh"
#include "../utils/cuda_defines.cuh"
#include "../utils/op.cuh"
#include "../utils/rng.cuh"
#include <stdint.h>

struct CudaScene;

enum LightType : uint8_t
{
    POINT = 0,
    CYLINDER,
    DIRECTIONNAL,
    QUAD,
    SUN,
    SPHERE_GEOM,
    IMPLICIT_SPHERE_GEOM,
    PLANE_GEOM,
    MESH_GEOM
};

struct alignas(16) Light
{
    //
    // PACKED DATA
    //

    // xyz = color
    // w   = intensity
    float4 color_power;

    // xyz = position
    // w   = radius
    float4 position_radius;

    // xyz = normal
    // w   = height
    float4 normal_height;

    // xyz = direction
    // w   = type
    float4 direction;

    // xyz = v
    // w   = geomIndex
    float3 v;

    uint32_t metadata;

    inline static uint32_t packMetadata(
        uint8_t type,
        uint32_t geomIndex)
    {
        return
            uint32_t(type)
            | (geomIndex << 8);
    }

    //
    // ACCESSORS
    //

    __device__ inline LightType getType() const
    {
        return LightType(uint8_t(metadata & 0xFF));
    }

    __device__ inline uint32_t getGeomIndex() const
    {
        return metadata >> 8;
    }

    __device__ inline float3 getColor() const
    {
        return make_float3(color_power);
    }

    __device__ inline float getIntensity() const
    {
        return color_power.w;
    }

    __device__ inline float3 getColorPower() const
    {
        return getColor() * getIntensity();
    }

    __device__ inline float3 getPosition() const
    {
        return make_float3(position_radius);
    }

    __device__ inline float getRadius() const
    {
        return position_radius.w;
    }

    __device__ inline float3 getNormal() const
    {
        return make_float3(normal_height);
    }

    __device__ inline float getHeight() const
    {
        return normal_height.w;
    }

    __device__ inline float3 getDirection() const
    {
        return make_float3(direction);
    }

    __device__ inline float3 getV() const
    {
        return v;
    }

    //
    // SAMPLING
    //

    __device__
    LightSample sampleSphereGeom(
        const float3& p_point,
        RNG* rng,
        const CudaScene& scene
    ) const;

    __device__
    LightSample sampleImplicitSphereGeom(
        const float3& p_point,
        RNG* rng,
        const CudaScene& scene
    ) const;

    __device__
    LightSample samplePlaneGeom(
        const float3& p_point,
        RNG* rng,
        const CudaScene& scene
    ) const;

    __device__
    LightSample sampleMeshGeom(
        const float3& p_point,
        RNG* rng,
        const CudaScene& scene
    ) const;

    __device__
    LightSample sampleCylinder(
        const float3& p_point,
        RNG* rng
    ) const;

    __device__
    LightSample sampleDirectionnal(
        const float3& p_point
    ) const;

    __device__
    LightSample samplePoint(
        const float3& p_point
    ) const;

    __device__
    LightSample sampleCone(
        const float3& p_point,
        RNG* rng
    ) const;

    __device__
    LightSample sampleQuad(
        const float3& p_point,
        RNG* rng
    ) const;

    __device__
    LightSample sample(
        const float3& p_point,
        RNG* rng,
        const CudaScene& scene
    ) const;

    __device__
    LightSample sample(
        const float3& p_point,
        RNG* rng
    ) const;

    __device__
    LightSample sample(
        const float3& p_point
    ) const;
};