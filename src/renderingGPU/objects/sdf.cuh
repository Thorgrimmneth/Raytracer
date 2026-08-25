#pragma once

#include "../utils/check.cuh"
#include "../utils/op.cuh"
#include "../utils/rng.cuh"
#include "../utils/quaternion.cuh"
#include "cone.cuh"
#include "csg_tree.cuh"
#include "sphere_analytic.cuh"
#include "signed_grid.cuh"
#include "tore.cuh"
#include <vector>

enum class SDFType : uint8_t
{
    SphereAnalytic,
    Tore,
    Cone,
    CSGTree,
    SignedGrid
};

struct SDF
{
    SDFType type;
    float3 translation = make_float3(0.f);
    Matrix3x3 rotation = Matrix3x3::identity();
    OptixAabb aabb;
    int lightIndex = -1; // Index into the lights array, if this instance is emissive
    union {
        SphereAnalytic sphere;
        Tore tore;
        Cone cone;
        CSGTree csgTree;
        SignedGrid signedGrid;
    };

    // Default constructor
    HD SDF() : type(SDFType::SphereAnalytic), sphere({0.f, 0}) {}

    DEVICE inline float sdf(const float3 &point) const
    {
        switch (type)
        {
        case SDFType::Tore:
            return tore.sdf(point);
        case SDFType::Cone:
            return cone.sdf(point);
        case SDFType::CSGTree:
            return csgTree.sdf(point);
        case SDFType::SignedGrid:
            return signedGrid.sdf(point);
        default:
            return 0.0f; // Should not happen
        }
    }

    HD int getMaterialIndex() const
    {
        switch (type)
        {
        case SDFType::SphereAnalytic:
            return sphere.materialIndex;
        case SDFType::Tore:
            return tore.materialIndex;
        case SDFType::Cone:
            return cone.materialIndex;
        case SDFType::CSGTree:
            return csgTree.materialIndex;
        case SDFType::SignedGrid:
            return signedGrid.materialIndex;
        default:
            return 0; // Should not happen
        }
    }

    HOST OptixAabb getWorldAABB(const std::vector<PrimitiveData> &primitives = {},
                                const std::vector<CSGNode> &nodes = {}) const
    {
        switch (type)
        {
        case SDFType::SphereAnalytic:
            return sphere.computeWorldAABB(translation);
        case SDFType::Tore:
            return tore.computeWorldAABB(rotation, translation);
        case SDFType::Cone:
            return cone.computeWorldAABB(rotation, translation);
        case SDFType::CSGTree:
            // Note: CSGTree requires full SDF array, use getWorldAABBWithContext() instead
            return csgTree.computeWorldAABB(rotation, translation, primitives, nodes);
        case SDFType::SignedGrid:
            return signedGrid.computeWorldAABB(rotation, translation);
        default:
            return OptixAabb(); // Should not happen
        }
    }

    DEVICE void samplePoint(float3 &p_point, float3 &p_normal, RNG &rng) const
    {
        switch (type)
        {
        case SDFType::SphereAnalytic:
            sphere.sampleSurfacePoint(p_point, p_normal, rng);
            break;
        case SDFType::Tore:
            tore.sampleSurfacePoint(p_point, p_normal, rng);
            break;
        case SDFType::Cone:
            cone.sampleSurfacePoint(p_point, p_normal, rng);
            break;
        case SDFType::CSGTree:
            csgTree.sampleSurfacePoint(p_point, p_normal, rng);
            break;
        case SDFType::SignedGrid:
            signedGrid.sampleSurfacePoint(p_point, p_normal, rng);
            break;
        default:
            break; // Should not happen
        }
    }

    HD float getArea() const
    {
        switch (type)
        {
        case SDFType::SphereAnalytic:
            return 4.f * M_PIf * sphere.radius * sphere.radius;
        case SDFType::Tore:
            return 4.f * M_PIf * M_PIf * tore.radiusExter * tore.radiusInter;
        case SDFType::CSGTree:
            return csgTree.getArea();
        case SDFType::Cone:
            return cone.getArea();
        case SDFType::SignedGrid:
            return signedGrid.getArea();
        default:
            return 0.f; // Should not happen
        }
    }

    HOST OptixAabb getAABB(const std::vector<PrimitiveData> &primitives = {},
                           const std::vector<CSGNode> &nodes = {}) const
    {
        switch (type)
        {
        case SDFType::SphereAnalytic:
            return sphere.computeAABB();
        case SDFType::Tore:
            return tore.computeAABB();
        case SDFType::Cone:
            return cone.computeAABB();
        case SDFType::CSGTree:
            return csgTree.computeAABB(primitives, nodes);
        case SDFType::SignedGrid:
            return signedGrid.computeAABB();
        default:
            return OptixAabb(); // Should not happen
        }
    }

    DEVICE float3 getNormal(const float3 &point) const
    {
        switch (type)
        {
        case SDFType::Tore:
            return tore.getNormal(point);
        case SDFType::Cone:
            return cone.getNormal(point);
        case SDFType::CSGTree:
            return csgTree.getNormal(point);
        case SDFType::SignedGrid:
            return signedGrid.getNormal(point);
        default:
            return make_float3(0.f); // Should not happen
        }
    }

    static SDF createRandomSphereAnalytic(int materialIndex, float radius = 0.f)
    {
        SDF sdf;
        sdf.type = SDFType::SphereAnalytic;
        if (radius <= 0.f)
            radius = randomFloat() * 1.f + 0.2f;
        sdf.sphere = SphereAnalytic::createRandomSphere(radius, materialIndex);
        sdf.aabb = sdf.sphere.computeAABB();
        sdf.translation =
            make_float3(randomFloat() * 10.f - 5.f, randomFloat() * 10.f - 5.f, randomFloat() * 10.f - 5.f);
        return sdf;
    }

    static SDF createRandomToreSDF(int materialIndex)
    {
        SDF sdf;
        sdf.type = SDFType::Tore;
        sdf.tore = Tore::createRandomTore(materialIndex);
        sdf.aabb = sdf.tore.computeAABB();
        sdf.translation =
            make_float3(randomFloat() * 10.f - 5.f, randomFloat() * 10.f - 5.f, randomFloat() * 10.f - 5.f);
        Quaternion rotation = quaternionFromAxisAngle(
            make_float3(randomFloat() * 2.f, randomFloat() * 2.f, randomFloat() * 2.f), randomFloat() * 360.f);
        Matrix3x3 rotationMatrix = quaternionToMatrix(rotation);
        sdf.rotation = rotationMatrix.transpose();
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
