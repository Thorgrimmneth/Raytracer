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

    inline OptixAabb getAABB() const
    {
        switch (type)
        {
        case SDFType::Sphere:
            return sphere.computeAABB();
        case SDFType::Tore:
            return tore.computeAABB(rotation, translation);
        default:
            return OptixAabb(); // Should not happen
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
