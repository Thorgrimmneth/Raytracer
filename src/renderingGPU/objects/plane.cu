#include "plane.cuh"
#include <stdio.h>

__device__ __forceinline__
bool Plane::intersectGeometry(const Ray& ray, float& t) const
{
    // Fast path pour le sol horizontal y = 0
    // Ton plan dans spheresScene :
    // normal = (0, 1, 0), delta = 0
    if (normal.x == 0.0f && normal.y == 1.0f && normal.z == 0.0f && delta == 0.0f)
    {
        const float dy = ray.direction.y;

        if (fabsf(dy) < 1e-6f)
            return false;

        t = -ray.origin.y / dy;
        return t >= 0.0f;
    }

    const float ND = dot(normal, ray.direction);

    if (fabsf(ND) < 1e-6f)
        return false;

    t = -(dot(normal, ray.origin) + delta) / ND;

    return t >= 0.0f;
}

__device__
bool Plane::intersect(
    const Ray& ray,
    const float tMin,
    const float tMax,
    HitRecord& hitRecord) const
{
    float t;

    if (!intersectGeometry(ray, t))
        return false;

    if (t <= tMin || t >= tMax)
        return false;

    hitRecord.point = ray.pointAtT(t);
    hitRecord.normal = normal;
    hitRecord.faceNormal(ray.direction);
    hitRecord.distance = t;
    hitRecord.materialIndex = materialIndex;

    return true;
}

