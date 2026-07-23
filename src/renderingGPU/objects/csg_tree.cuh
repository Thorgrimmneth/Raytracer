#pragma once

#pragma once

#include "../../utils/rng_cpu.hpp"
#include "../utils/op.cuh"
#include "../utils/rng.cuh"
#include "sphere.cuh"
#include "tore.cuh"
#include <optix.h>
#include <optix_stubs.h>

#include <vector>

enum class PrimitiveType : uint8_t
{
    Sphere,
    Tore
};

struct PrimitiveData
{
    PrimitiveType type;
    float3 translation = make_float3(0.f);
    Matrix3x3 rotation = Matrix3x3::identity();
    union {
        Sphere sphere;
        Tore torus;
    };

    __device__ inline float sdf(const float3 &p) const
    {
        // Transform point from world space to primitive local space
        float3 pLocal = transform(rotation, p - translation);

        switch (type)
        {
        case PrimitiveType::Sphere:
            return sphere.sdf(pLocal);

        case PrimitiveType::Tore:
            return torus.sdf(pLocal);
        }

        return 1e20f;
    }

    H_INLINE OptixAabb computeWorldAABB(const Matrix3x3 &p_rotation, const float3 &p_translation) const
    {
        switch (type)
        {
        case PrimitiveType::Sphere:
            return sphere.computeWorldAABB(p_translation);

        case PrimitiveType::Tore:
            return torus.computeWorldAABB(p_rotation, p_translation);
        }
        return OptixAabb();
    }

    HD_INLINE OptixAabb computeAABB() const
    {
        switch (type)
        {
        case PrimitiveType::Sphere:
            return sphere.computeAABB();

        case PrimitiveType::Tore:
            return torus.computeAABB();
        }

        return OptixAabb();
    }
    static PrimitiveData createSpherePrimitive(const float3 &translation, int materialIndex)
    {
        PrimitiveData pd;
        pd.type = PrimitiveType::Sphere;
        pd.translation = translation;
        pd.sphere = Sphere::createRandomSphere(materialIndex);
        return pd;
    }

    static PrimitiveData createTorePrimitive(const Matrix3x3 &rotation, const float3 &translation, int materialIndex)
    {
        PrimitiveData pd;
        pd.type = PrimitiveType::Tore;
        pd.rotation = rotation;
        pd.translation = translation;
        pd.torus = Tore::createRandomTore(materialIndex);
        return pd;
    }
};

enum class CSGOp : uint8_t
{
    Union,
    Intersection,
    Difference
};

struct CSGNode
{
    CSGOp operation;
    uint32_t left;  // 4 bytes (bit 31 = leaf flag, bits 0-30 = left index)
    uint32_t right; // 4 bytes (bit 31 = leaf flag, bits 0-30 = right index)

    CSGNode() : operation(CSGOp::Union), left(0), right(0) {}
    CSGNode(CSGOp op, uint32_t leftIndex, uint32_t rightIndex) : operation(op), left(leftIndex), right(rightIndex) {}
    CSGNode(uint32_t leftIndex, uint32_t rightIndex) : operation(CSGOp::Union), left(leftIndex), right(rightIndex) {}

    HD_FORCEINLINE bool leftIsLeaf() const { return (left & 0x80000000u) != 0; }

    HD_FORCEINLINE bool rightIsLeaf() const { return (right & 0x80000000u) != 0; }
    HD_FORCEINLINE uint32_t getLeftIndex() const { return left & 0x7FFFFFFFu; }
    HD_FORCEINLINE uint32_t getRightIndex() const { return right & 0x7FFFFFFFu; }
};

struct Frame
{
    uint32_t node;
    uint8_t state;

    float leftValue;
    float rightValue;
};

struct CSGTree
{
    int materialIndex;
    PrimitiveData *primArray;
    CSGNode *nodes;

    int nbNodes;
    float area = 0.f;

    __device__ float sdf(const float3 &point) const
    {
        Frame stack[64];
        int sp = 0;

        stack[sp++] = {0, 0, 0.f, 0.f}; // root node

        float lastValue = 0.f;

        while (sp)
        {
            Frame &f = stack[sp - 1];
            const CSGNode &node = nodes[f.node];

            switch (f.state)
            {
            //-----------------------------------------
            // Evaluate left child
            //-----------------------------------------
            case 0: {
                if (node.leftIsLeaf())
                {
                    f.leftValue = primArray[node.getLeftIndex()].sdf(point);
                    f.state = 1;
                }
                else
                {
                    f.state = 1;
                    stack[sp++] = {node.getLeftIndex(), 0, 0.f, 0.f};
                }
                break;
            }

            //-----------------------------------------
            // Evaluate right child
            //-----------------------------------------
            case 1: {
                // If left child was a node, retrieve its result
                if (!node.leftIsLeaf())
                    f.leftValue = lastValue;

                if (node.rightIsLeaf())
                {
                    f.rightValue = primArray[node.getRightIndex()].sdf(point);
                    f.state = 2;
                }
                else
                {
                    f.state = 2;
                    stack[sp++] = {node.getRightIndex(), 0, 0.f, 0.f};
                }
                break;
            }

            //-----------------------------------------
            // Combine and return to parent
            //-----------------------------------------
            case 2: {
                if (!node.rightIsLeaf())
                    f.rightValue = lastValue;

                switch (node.operation)
                {
                case CSGOp::Union:
                    lastValue = fminf(f.leftValue, f.rightValue);
                    break;

                case CSGOp::Intersection:
                    lastValue = fmaxf(f.leftValue, f.rightValue);
                    break;

                case CSGOp::Difference:
                    lastValue = fmaxf(f.leftValue, -f.rightValue);
                    break;
                }

                --sp;
                break;
            }
            }
        }

        return lastValue;
    }

    H_INLINE OptixAabb computeWorldAABB(const Matrix3x3 &rotation, const float3 &translation,
                                        const std::vector<PrimitiveData> &primitives,
                                        const std::vector<CSGNode> &nodesCPU) const
    {
        OptixAabb aabb;
        aabb.minX = 1e20f;
        aabb.minY = 1e20f;
        aabb.minZ = 1e20f;
        aabb.maxX = -1e20f;
        aabb.maxY = -1e20f;
        aabb.maxZ = -1e20f;
        for (int i = 0; i < nbNodes; ++i)
        {
            const CSGNode &node = nodesCPU[i];

            if (node.leftIsLeaf())
            {
                const PrimitiveData &prim = primitives[node.getLeftIndex()];
                Matrix3x3 worldRotation = rotation * prim.rotation;
                float3 worldTranslation = translation + rotation * prim.translation;
                OptixAabb primAabb = prim.computeWorldAABB(worldRotation, worldTranslation);
                aabb.minX = fminf(aabb.minX, primAabb.minX);
                aabb.minY = fminf(aabb.minY, primAabb.minY);
                aabb.minZ = fminf(aabb.minZ, primAabb.minZ);
                aabb.maxX = fmaxf(aabb.maxX, primAabb.maxX);
                aabb.maxY = fmaxf(aabb.maxY, primAabb.maxY);
                aabb.maxZ = fmaxf(aabb.maxZ, primAabb.maxZ);
            }

            if (node.rightIsLeaf())
            {
                const PrimitiveData &prim = primitives[node.getRightIndex()];
                Matrix3x3 worldRotation = rotation * prim.rotation;
                float3 worldTranslation = translation + rotation * prim.translation;
                OptixAabb primAabb = prim.computeWorldAABB(worldRotation, worldTranslation);
                aabb.minX = fminf(aabb.minX, primAabb.minX);
                aabb.minY = fminf(aabb.minY, primAabb.minY);
                aabb.minZ = fminf(aabb.minZ, primAabb.minZ);
                aabb.maxX = fmaxf(aabb.maxX, primAabb.maxX);
                aabb.maxY = fmaxf(aabb.maxY, primAabb.maxY);
                aabb.maxZ = fmaxf(aabb.maxZ, primAabb.maxZ);
            }
        }
        return aabb;
    }

    HOST OptixAabb computeAABB(const std::vector<PrimitiveData> &primitives, const std::vector<CSGNode> &nodesCPU) const
    {
        OptixAabb aabb;
        aabb.minX = 1e20f;
        aabb.minY = 1e20f;
        aabb.minZ = 1e20f;
        aabb.maxX = -1e20f;
        aabb.maxY = -1e20f;
        aabb.maxZ = -1e20f;
        for (int i = 0; i < nbNodes; ++i)
        {
            const CSGNode &node = nodesCPU[i];

            if (node.leftIsLeaf())
            {
                const PrimitiveData &prim = primitives[node.getLeftIndex()];
                OptixAabb primAabb = prim.computeWorldAABB(prim.rotation, prim.translation);
                aabb.minX = fminf(aabb.minX, primAabb.minX);
                aabb.minY = fminf(aabb.minY, primAabb.minY);
                aabb.minZ = fminf(aabb.minZ, primAabb.minZ);
                aabb.maxX = fmaxf(aabb.maxX, primAabb.maxX);
                aabb.maxY = fmaxf(aabb.maxY, primAabb.maxY);
                aabb.maxZ = fmaxf(aabb.maxZ, primAabb.maxZ);
            }

            if (node.rightIsLeaf())
            {
                const PrimitiveData &prim = primitives[node.getRightIndex()];
                OptixAabb primAabb = prim.computeWorldAABB(prim.rotation, prim.translation);
                aabb.minX = fminf(aabb.minX, primAabb.minX);
                aabb.minY = fminf(aabb.minY, primAabb.minY);
                aabb.minZ = fminf(aabb.minZ, primAabb.minZ);
                aabb.maxX = fmaxf(aabb.maxX, primAabb.maxX);
                aabb.maxY = fmaxf(aabb.maxY, primAabb.maxY);
                aabb.maxZ = fmaxf(aabb.maxZ, primAabb.maxZ);
            }
        }
        return aabb;
    }

    __device__ float3 sampleSurfacePoint(const float3 &p_point, const float3 &p_normal, const RNG &rng) const
    {
        return make_float3(0.0f);
    }

    __device__ float3 getNormal(const float3 &point) const
    {
        float h = 1e-4f;
        float3 normal = make_float3(sdf(point + make_float3(h, 0.0f, 0.0f)) - sdf(point - make_float3(h, 0.0f, 0.0f)),
                                    sdf(point + make_float3(0.0f, h, 0.0f)) - sdf(point - make_float3(0.0f, h, 0.0f)),
                                    sdf(point + make_float3(0.0f, 0.0f, h)) - sdf(point - make_float3(0.0f, 0.0f, h)));
        return normalize(normal);
    }

    __device__ float getArea() const { return area; }
};