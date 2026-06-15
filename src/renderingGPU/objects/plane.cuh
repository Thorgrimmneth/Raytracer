#pragma once

#include "../materials/material.cuh"
#include "../../../devicePrograms/launch_params.cuh"
struct Plane
{
    float4 normal; // xyz = normal, w = delta
    int materialIndex;

    Plane() = default;
    Plane(float3 pos, float3 n) : normal(make_float4(n, 0)), materialIndex(0) {}

    D_FORCEINLINE
    float3 getNormal() const { return make_float3(normal); }

    D_FORCEINLINE
    float getDelta() const { return normal.w; }

    D_FORCEINLINE 
    bool intersect(const float4 &origin, const float4 &direction, const float tMin, const float tMax, OptixHit &hitRecord) const
    {
        float t;

        // Fast path pour le sol horizontal y = 0
        if (normal.x == 0.0f && normal.y == 1.0f && normal.z == 0.0f && getDelta() == 0.0f)
        {
            const float dy = direction.y;

            if (fabsf(dy) < 1e-6f)
                return false;

            t = -origin.y / dy;
        }
        else
        {
            const float nd = dot(getNormal(), direction);

            if (fabsf(nd) < 1e-6f)
                return false;

            t = -(dot(getNormal(), origin) + getDelta()) / nd;
        }

        if (t <= tMin || t >= tMax)
            return false;

        const float4 p = origin + t * direction;
        float3 n = getNormal();

        hitRecord.setHitInfo(p, n, t, materialIndex, 0, HIT_PLANE);
        hitRecord.faceNormal(direction);
        return true;
    }

    D_FORCEINLINE
    bool intersectAny(const float4 &origin, const float4 &direction, const float tMin, const float tMax,
                                                 const Material *materials) const
    {
        if (materials[materialIndex].type() == MaterialType::TRANSPARENT)
            return false;

        // Fast path ultra-court pour sol horizontal y = 0
        float3 n = getNormal();
        if (n.x == 0.0f && n.y == 1.0f && n.z == 0.0f && getDelta() == 0.0f)
        {
            const float oy = origin.y;
            const float dy = direction.y;

            // Rayon parallèle au sol
            if (fabsf(dy) < 1e-6f)
                return false;

            // Si on est au-dessus du sol et qu'on part vers le haut,
            // impossible de toucher y = 0.
            if (oy > 0.0f && dy >= 0.0f)
                return false;

            const float t = -oy / dy;

            return t > tMin && t < tMax;
        }

        const float ND = dot(getNormal(), direction);

        if (fabsf(ND) < 1e-6f)
            return false;

        const float t = -(dot(getNormal(), origin) + getDelta()) / ND;

        return t > tMin && t < tMax;
    }
};