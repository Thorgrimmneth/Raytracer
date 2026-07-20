#pragma once

#include "../utils/check.cuh"
#include "../utils/op.cuh"
#include "sphere.cuh"
#include "tore.cuh"

enum class SDFType : uint8_t
{
    Sphere,
    Tore
};

struct SDF
{
    SDFType type;
    float3 translation = make_float3(0.f);
    Matrix3x3 rotation = Matrix3x3::identity();
    int lightIndex = -1; // Index into the lights array, if this instance is emissive
    union {
        Sphere sphere;
        Tore tore;
    };

    __device__ float sdf(const float3 &point) const
    {
        switch (type)
        {
        case SDFType::Sphere:
            return sphere.sdf(point);
        case SDFType::Tore:
            return tore.sdf(point);
        default:
            return 0.0f; // Should not happen
        }
    }

    __device__ int getMaterialIndex() const
    {
        switch (type)
        {
        case SDFType::Sphere:
            return sphere.materialIndex;
        case SDFType::Tore:
            return tore.materialIndex;
        default:
            return 0; // Should not happen
        }
    }

    inline OptixAabb getWorldAABB() const
    {
        switch (type)
        {
        case SDFType::Sphere:
            return sphere.computeWorldAABB(translation);
        case SDFType::Tore:
            return tore.computeWorldAABB(rotation, translation);
        default:
            return OptixAabb(); // Should not happen
        }
    }

    D_FORCEINLINE void samplePoint(float3 &p_point, float3 &p_normal, RNG &rng) const
    {
        switch (type)
        {
        case SDFType::Sphere:
            sphere.sampleSurfacePoint(p_point, p_normal, rng);
            break;
        case SDFType::Tore:
            tore.sampleSurfacePoint(p_point, p_normal, rng);
            break;
        default:
            break; // Should not happen
        }
    }

    D_FORCEINLINE float getArea() const
    {
        switch (type)
        {
        case SDFType::Sphere:
            return 4.f * M_PIf * sphere.radius * sphere.radius;
        case SDFType::Tore:
            return 4.f * M_PIf * M_PIf * tore.radiusExter * tore.radiusInter;
        default:
            return 0.f; // Should not happen
        }
    }
    
    HD_FORCEINLINE OptixAabb getAABB() const
    {
        switch (type)
        {
        case SDFType::Sphere:
            return sphere.computeAABB();
        case SDFType::Tore:
            return tore.computeAABB(rotation);
        default:
            return OptixAabb(); // Should not happen
        }
    }

    D_FORCEINLINE float3 getNormal(const float3 &point) const
    {
        switch (type)
        {
        case SDFType::Sphere:
            return sphere.getNormal(point);
        case SDFType::Tore:
            return tore.getNormal(point, rotation);
        default:
            return make_float3(0.f); // Should not happen
        }
    }

    static SDF createRandomSphereSDF(int materialIndex)
    {
        SDF sdf;
        sdf.type = SDFType::Sphere;
        sdf.sphere = Sphere::createRandomSphere(materialIndex);
        return sdf;
    }

    static SDF createRandomToreSDF(int materialIndex)
    {
        SDF sdf;
        sdf.type = SDFType::Tore;
        sdf.tore = Tore::createRandomTore(materialIndex);
        return sdf;
    }
};

struct SDFGeometry
{
    SDF *sdfs = nullptr;
    CUdeviceptr d_aabbBuffer = 0;
    int sdfCount = 0;
};

inline SDFGeometry loadSDFGeometry(const std::vector<SDF> &sdfs)
{
    SDFGeometry geometry;
    geometry.sdfCount = static_cast<int>(sdfs.size());
    if (!sdfs.empty())
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&geometry.sdfs), sizeof(SDF) * sdfs.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(geometry.sdfs), sdfs.data(), sizeof(SDF) * sdfs.size(),
                              cudaMemcpyHostToDevice));
    }
    return geometry;
}

HD_INLINE bool intersectAABB(const float3 &origin, const float3 &dir, const OptixAabb &box, float &tMin, float &tMax)
{
    float3 invDir = make_float3(1.f / dir.x, 1.f / dir.y, 1.f / dir.z);

    float3 t0 = (make_float3(box.minX, box.minY, box.minZ) - origin) * invDir;
    float3 t1 = (make_float3(box.maxX, box.maxY, box.maxZ) - origin) * invDir;

    float3 tNear = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));

    float3 tFar = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));

    tMin = fmaxf(tMin, fmaxf(fmaxf(tNear.x, tNear.y), tNear.z));
    tMax = fminf(tMax, fminf(fminf(tFar.x, tFar.y), tFar.z));

    return tMin <= tMax;
}
