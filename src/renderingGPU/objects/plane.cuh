#pragma once

#include "../utils/op.cuh"
#include "../raytracingUtils/ray.cuh"
#include "../raytracingUtils/hitrecord.cuh"
#include "../materials/material.cuh"

struct Plane
{
	float3 normal;
    float delta;
    int materialIndex;

	Plane() = default;
	Plane(float3 pos, float3 n) : normal(n), delta(dot(-n, pos)) {}
	__device__
    bool intersectGeometry( const Ray & ray, float & p_t1 ) const;

	__device__
	bool intersect( const Ray & p_ray, const float p_tMin, const float p_tMax, HitRecord & p_hitRecord ) const;

	__device__ __forceinline__
bool intersectAny(
    const Ray& ray,
    const float tMin,
    const float tMax,
    const Material* materials) const
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