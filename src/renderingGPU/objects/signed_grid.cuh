#pragma once

#include "../../utils/rng_cpu.hpp"
#include "../utils/op.cuh"
#include "../utils/rng.cuh"
#include <optix.h>
#include <optix_stubs.h>

struct SignedGrid
{
    float *data = nullptr;   // one float per voxel
    uint32_t resolution = 0; // number of voxel per dimension
    int materialIndex = 0;

    OptixAabb computeAABB() const
    {
        OptixAabb aabb;
        aabb.minX = -1.f;
        aabb.minY = -1.f;
        aabb.minZ = -1.f;
        aabb.maxX = 1.f;
        aabb.maxY = 1.f;
        aabb.maxZ = 1.f;
        return aabb;
    }

    OptixAabb computeWorldAABB(const Matrix3x3 &rotation, const float3 &translation) const
    {
        OptixAabb localAABB = computeAABB();
        float3 corners[8] = {make_float3(localAABB.minX, localAABB.minY, localAABB.minZ),
                             make_float3(localAABB.minX, localAABB.minY, localAABB.maxZ),
                             make_float3(localAABB.minX, localAABB.maxY, localAABB.minZ),
                             make_float3(localAABB.minX, localAABB.maxY, localAABB.maxZ),
                             make_float3(localAABB.maxX, localAABB.minY, localAABB.minZ),
                             make_float3(localAABB.maxX, localAABB.minY, localAABB.maxZ),
                             make_float3(localAABB.maxX, localAABB.maxY, localAABB.minZ),
                             make_float3(localAABB.maxX, localAABB.maxY, localAABB.maxZ)};

        OptixAabb worldAABB;
        worldAABB.minX = worldAABB.minY = worldAABB.minZ = std::numeric_limits<float>::max();
        worldAABB.maxX = worldAABB.maxY = worldAABB.maxZ = std::numeric_limits<float>::lowest();

        for (const auto &corner : corners)
        {
            float3 rotatedCorner = rotation * corner;
            float3 transformedCorner = rotatedCorner + translation;

            worldAABB.minX = fminf(worldAABB.minX, transformedCorner.x);
            worldAABB.minY = fminf(worldAABB.minY, transformedCorner.y);
            worldAABB.minZ = fminf(worldAABB.minZ, transformedCorner.z);

            worldAABB.maxX = fmaxf(worldAABB.maxX, transformedCorner.x);
            worldAABB.maxY = fmaxf(worldAABB.maxY, transformedCorner.y);
            worldAABB.maxZ = fmaxf(worldAABB.maxZ, transformedCorner.z);
        }

        return worldAABB;
    }

    __device__ inline float sdf(const float3 &point) const
    {
        // Convertir [-1, 1] vers [0, resolution - 1]
        float3 gridPos = (point + make_float3(1.0f)) * (0.5f * (resolution - 1));

        // Limiter à la grille
        gridPos.x = fmaxf(0.0f, fminf((float)(resolution - 1), gridPos.x));
        gridPos.y = fmaxf(0.0f, fminf((float)(resolution - 1), gridPos.y));
        gridPos.z = fmaxf(0.0f, fminf((float)(resolution - 1), gridPos.z));

        int x0 = floorf(gridPos.x);
        int y0 = floorf(gridPos.y);
        int z0 = floorf(gridPos.z);

        int x1 = fminf(x0 + 1, (int)resolution - 1);
        int y1 = fminf(y0 + 1, (int)resolution - 1);
        int z1 = fminf(z0 + 1, (int)resolution - 1);

        float tx = gridPos.x - x0;
        float ty = gridPos.y - y0;
        float tz = gridPos.z - z0;

        auto index = [this](int x, int y, int z) {
            return x * resolution * resolution + y * resolution + z;
        };

        float c000 = data[index(x0, y0, z0)];
        float c100 = data[index(x1, y0, z0)];
        float c010 = data[index(x0, y1, z0)];
        float c110 = data[index(x1, y1, z0)];

        float c001 = data[index(x0, y0, z1)];
        float c101 = data[index(x1, y0, z1)];
        float c011 = data[index(x0, y1, z1)];
        float c111 = data[index(x1, y1, z1)];

        // interpolation X
        float c00 = lerp(c000, c100, tx);
        float c10 = lerp(c010, c110, tx);
        float c01 = lerp(c001, c101, tx);
        float c11 = lerp(c011, c111, tx);

        // interpolation Y
        float c0 = lerp(c00, c10, ty);
        float c1 = lerp(c01, c11, ty);

        // interpolation Z
        return lerp(c0, c1, tz);
    }

    __device__ float3 sampleSurfacePoint(const float3 &p_point, const float3 &p_normal, const RNG &rng) const
    {
        return make_float3(0.0f);
    }

    __device__ float3 getNormal(const float3 &point) const
    {
        float h = 0.5f * (2.0f / resolution);
        float3 normal = make_float3(sdf(point + make_float3(h, 0.0f, 0.0f)) - sdf(point - make_float3(h, 0.0f, 0.0f)),
                                    sdf(point + make_float3(0.0f, h, 0.0f)) - sdf(point - make_float3(0.0f, h, 0.0f)),
                                    sdf(point + make_float3(0.0f, 0.0f, h)) - sdf(point - make_float3(0.0f, 0.0f, h)));
        return normalize(normal);
    }

    __device__ float getArea() const { return 1.f; }
};
