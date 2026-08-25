#pragma once

#include "../../utils/rng_cpu.hpp"
#include <optix.h>
#include <optix_stubs.h>

struct Cone
{
    float height;
    float radius;
    float angle;
    float area;
    int materialIndex;

    static Cone createRandomCone(int materialIndex)
    {
        float height = randomFloat() * 2.f + 0.2f;
        float angle = randomFloat() * 30.f + 10.f;
        return create(height, angle, materialIndex);
    }

    static Cone create(float h, float angleDeg, int m)
    {
        float angle = angleDeg * M_PIf / 180.0f;
        float radius = h * tanf(angle);
        float slant = sqrt(h * h + radius * radius);
        float sideArea = M_PIf * radius * slant;
        float totalArea = sideArea + M_PIf * radius * radius;
        return {h, radius, angle, totalArea, m};
    }

    H_INLINE OptixAabb computeWorldAABB(const Matrix3x3 &rotation, const float3 &translation) const
    {
        float3 corners[8] = {make_float3(-radius, -height, -radius), make_float3(-radius, -height, radius),
                             make_float3(-radius, 0.0f, -radius),    make_float3(-radius, 0.0f, radius),

                             make_float3(radius, -height, -radius),  make_float3(radius, -height, radius),
                             make_float3(radius, 0.0f, -radius),     make_float3(radius, 0.0f, radius)};

        Matrix3x3 localToWorld = rotation.transpose();

        float3 p = transform(localToWorld, corners[0]) + translation;

        float3 minPoint = p;
        float3 maxPoint = p;

        for (int i = 1; i < 8; ++i)
        {
            p = transform(localToWorld, corners[i]) + translation;

            minPoint.x = fminf(minPoint.x, p.x);
            minPoint.y = fminf(minPoint.y, p.y);
            minPoint.z = fminf(minPoint.z, p.z);

            maxPoint.x = fmaxf(maxPoint.x, p.x);
            maxPoint.y = fmaxf(maxPoint.y, p.y);
            maxPoint.z = fmaxf(maxPoint.z, p.z);
        }

        OptixAabb aabb;
        aabb.minX = minPoint.x;
        aabb.minY = minPoint.y;
        aabb.minZ = minPoint.z;

        aabb.maxX = maxPoint.x;
        aabb.maxY = maxPoint.y;
        aabb.maxZ = maxPoint.z;

        return aabb;
    }

    D_FORCEINLINE void sampleSurfacePoint(float3 &p, float3 &n, RNG &rng) const
    {
        float slant = sqrtf(radius * radius + height * height);

        float sideArea = M_PIf * radius * slant;
        float baseArea = M_PIf * radius * radius;

        if (rng.nextFloat() < sideArea / (sideArea + baseArea))
        {
            float u = rng.nextFloat();
            float v = rng.nextFloat();

            float rho = radius * sqrtf(u);
            float theta = 2.0f * M_PIf * v;

            float x = rho * cosf(theta);
            float z = rho * sinf(theta);
            float y = -height * rho / radius;

            p = make_float3(x, y, z);

            n = normalize(make_float3(x, radius * radius / height, z));
        }
        else
        {
            float r = radius * sqrtf(rng.nextFloat());
            float theta = 2.0f * M_PIf * rng.nextFloat();

            p = make_float3(r * cosf(theta), -height, r * sinf(theta));

            n = make_float3(0.f, -1.f, 0.f);
        }
    }

    HD_FORCEINLINE OptixAabb computeAABB() const
    {
        OptixAabb aabb;
        aabb.minX = -radius;
        aabb.maxX = radius;

        aabb.minY = -height;
        aabb.maxY = 0.0f;

        aabb.minZ = -radius;
        aabb.maxZ = radius;

        return aabb;
    }

    D_FORCEINLINE float3 getNormal(const float3 &point) const
    {
        float eps = 1e-4f;

        float3 n = normalize(make_float3(sdf(point + make_float3(eps, 0, 0)) - sdf(point - make_float3(eps, 0, 0)),
                                         sdf(point + make_float3(0, eps, 0)) - sdf(point - make_float3(0, eps, 0)),
                                         sdf(point + make_float3(0, 0, eps)) - sdf(point - make_float3(0, 0, eps))));
        return n;
    }

    HD_FORCEINLINE float getArea() const { return area; }

    __device__ float sdf(const float3 &point) const
    {
        float2 q = make_float2(radius, -height);
        float2 w = make_float2(length(make_float2(point.x, point.z)), point.y);
        float2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0f, 1.0f);
        float2 b = w - q * make_float2(clamp(w.x / q.x, 0.0f, 1.0f), 1.0f);
        float k = (q.y < 0.0f) ? -1.0f : 1.0f;
        float d = fminf(dot(a, a), dot(b, b));
        float s = fmaxf(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
        return sqrtf(d) * sign(s);
    }
};