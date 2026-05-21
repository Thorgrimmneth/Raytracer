#pragma once

#include "../materials/material.cuh"
#include "../raytracingUtils/hitrecord.cuh"
#include "../raytracingUtils/ray.cuh"
#include "../utils/macro.cuh"
#include "../utils/op.cuh"

struct Plane
{
    float3 normal;
    float delta;
    int materialIndex;

    Plane() = default;
    Plane(float3 pos, float3 n) : normal(n), delta(dot(-n, pos)) {}

    D_FORCEINLINE 
    bool intersect(const Ray &ray, const float tMin, const float tMax, HitRecord &hitRecord) const
    {
        float t;

        // Fast path pour le sol horizontal y = 0
        if (normal.x == 0.0f && normal.y == 1.0f && normal.z == 0.0f && delta == 0.0f)
        {
            const float dy = ray.direction.y;

            if (fabsf(dy) < 1e-6f)
                return false;

            t = -ray.origin.y / dy;
        }
        else
        {
            const float nd = dot(normal, ray.direction);

            if (fabsf(nd) < 1e-6f)
                return false;

            t = -(dot(normal, ray.origin) + delta) / nd;
        }

        if (t <= tMin || t >= tMax)
            return false;

        const float3 p = ray.origin + t * ray.direction;
        float3 n = normal;

        hitRecord.point = p;
        hitRecord.normal = n;
        hitRecord.faceNormal(ray.direction);
        hitRecord.distance = t;
        hitRecord.materialIndex = materialIndex;

        return true;
    }

    D_FORCEINLINE
    bool intersectAny(const Ray &ray, const float tMin, const float tMax,
                                                 const Material *materials) const
    {
        if (materials[materialIndex].type() == MaterialType::TRANSPARENT)
            return false;

        // Fast path ultra-court pour sol horizontal y = 0
        if (normal.x == 0.0f && normal.y == 1.0f && normal.z == 0.0f && delta == 0.0f)
        {
            const float oy = ray.origin.y;
            const float dy = ray.direction.y;

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

        const float ND = dot(normal, ray.direction);

        if (fabsf(ND) < 1e-6f)
            return false;

        const float t = -(dot(normal, ray.origin) + delta) / ND;

        return t > tMin && t < tMax;
    }
};