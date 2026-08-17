#pragma once

#include <stdint.h>

#include "../materials/material.cuh"
#include "../objects/sdf.cuh"
#include "../objects/triangle_mesh.cuh"
#include "../utils/defines.cuh"
#include "../utils/op.cuh"
#include "../utils/rng.cuh"
#include "../utils/simplified_def.cuh"
#include "light_context.cuh"

#include "lightsample.cuh"

enum LightType : uint8_t
{
    POINT = 0,
    CYLINDER,
    DIRECTIONNAL,
    QUAD,
    SUN,
    SDF_GEOM,
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
    float4 vSample;

    uint32_t metadata;

    inline static uint32_t packMetadata(uint8_t type, uint32_t meshInstanceIndex)
    {
        return uint32_t(type) | (meshInstanceIndex << 8);
    }

    //
    // ACCESSORS
    //

    HD_FORCEINLINE LightType getType() const { return LightType(uint8_t(metadata & 0xFF)); }

    D_FORCEINLINE uint32_t getMeshInstanceIndex() const { return metadata >> 8; }
    D_FORCEINLINE uint32_t getSDFIndex() const { return metadata >> 8; }

    HD_FORCEINLINE float3 getColor() const { return make_float3(color_power); }

    HD_FORCEINLINE float getIntensity() const { return color_power.w; }

    HD_FORCEINLINE float3 getColorPower() const { return getColor() * getIntensity(); }

    D_FORCEINLINE float3 getPosition() const { return make_float3(position_radius); }

    D_FORCEINLINE float getRadius() const { return position_radius.w; }

    D_FORCEINLINE float3 getNormal() const { return make_float3(normal_height); }

    D_FORCEINLINE float getHeight() const { return normal_height.w; }

    D_FORCEINLINE float3 getDirection() const { return make_float3(direction); }

    D_FORCEINLINE float3 getV() const { return make_float3(vSample); }

    //
    // SAMPLING
    //

    DEVICE LightSample sampleSphereGeom(const float3 &p_point, RNG &rng, const LightContext &ctx) const;

    DEVICE LightSample samplePlaneGeom(const float3 &p_point, RNG &rng, const LightContext &ctx) const;

    DEVICE LightSample sampleSDFGeom(const float3 &p_point, RNG &rng, const SDF *scene_sdfs,
                                     const Material *scene_materials) const;

    DEVICE LightSample sampleMeshGeom(const float3 &p_point, RNG &rng, const MeshInstance *scene_mesh_instances,
                                      const MeshGeometry *scene_mesh_geometries, const Material *scene_materials) const;

    DEVICE LightSample sampleCylinder(const float3 &p_point, RNG &rng) const;

    DEVICE LightSample sampleDirectionnal(const float3 &p_point) const;

    DEVICE LightSample samplePoint(const float3 &p_point) const;

    DEVICE LightSample sampleCone(const float3 &p_point, RNG &rng) const;

    DEVICE LightSample sampleQuad(const float3 &p_point, RNG &rng) const;

    DEVICE LightSample sample(const float3 &p_point, RNG &rng, const LightContext &ctx) const;

    DEVICE LightSample sample(const float3 &p_point, RNG &rng) const;

    DEVICE LightSample sample(const float3 &p_point) const;
};