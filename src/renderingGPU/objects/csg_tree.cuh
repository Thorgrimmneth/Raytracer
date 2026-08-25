#pragma once

#pragma once

#include "../../utils/rng_cpu.hpp"
#include "../utils/simplified_def.cuh"
#include "../utils/op.cuh"
#include "../utils/rng.cuh"
#include "cone.cuh"
#include "sphere_sdf.cuh"
#include "tore.cuh"
#include <optix.h>
#include <optix_stubs.h>

#include <vector>

enum class PrimitiveType : uint8_t
{
    SphereSDF,
    Tore,
    Cone
};

struct PrimitiveData
{
    PrimitiveType type;
    float3 translation = make_float3(0.f);
    Matrix3x3 rotation = Matrix3x3::identity();
    union {
        SphereSDF sphere;
        Tore torus;
        Cone cone;
    };

    __device__ inline float sdf(const float3 &p) const
    {
        // Transform point from world space to primitive local space
        float3 pLocal = transform(rotation, p - translation);

        switch (type)
        {
        case PrimitiveType::SphereSDF:
            return sphere.sdf(pLocal);
        case PrimitiveType::Tore:
            return torus.sdf(pLocal);
        case PrimitiveType::Cone:
            return cone.sdf(pLocal);
        }

        return 20000.f;
    }

    H_INLINE OptixAabb computeWorldAABB(const Matrix3x3 &p_rotation, const float3 &p_translation) const
    {
        switch (type)
        {
        case PrimitiveType::SphereSDF:
            return sphere.computeWorldAABB(p_translation);
        case PrimitiveType::Tore:
            return torus.computeWorldAABB(p_rotation, p_translation);
        case PrimitiveType::Cone:
            return cone.computeWorldAABB(p_rotation, p_translation);
        }
        return OptixAabb();
    }

    HD_INLINE OptixAabb computeAABB() const
    {
        switch (type)
        {
        case PrimitiveType::SphereSDF:
            return sphere.computeAABB();

        case PrimitiveType::Tore:
            return torus.computeAABB();
        case PrimitiveType::Cone:
            return cone.computeAABB();
        }
        return OptixAabb();
    }
    static PrimitiveData createSpherePrimitive(const float3 &translation, int materialIndex, float radius = -1.f)
    {
        PrimitiveData pd;
        pd.type = PrimitiveType::SphereSDF;
        pd.translation = translation;
        if (radius < 0.f)
            pd.sphere = SphereSDF::createRandomSphere(materialIndex);
        else
            pd.sphere = SphereSDF::create(radius, materialIndex);
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

    static PrimitiveData createConePrimitive(const Matrix3x3 &rotation, const float3 &translation, int materialIndex,
                                             float height = -1.f, float angleDeg = -1.f)
    {
        PrimitiveData pd;
        pd.type = PrimitiveType::Cone;
        pd.rotation = rotation;
        pd.translation = translation;
        if (height < 0.f || angleDeg < 0.f)
            pd.cone = Cone::createRandomCone(materialIndex);
        else
            pd.cone = Cone::create(height, angleDeg, materialIndex);
        return pd;
    }
};

enum class InstructionOp : uint8_t
{
    Primitive, // only used for leaf
    Union,
    Intersection,
    Difference,
    SmoothUnion,
    SmoothIntersection,
    SmoothDifference
};

D_FORCEINLINE float smoothUnion(float d1, float d2, float k)
{
    k *= 4.f;
    float h = fmaxf(k - fabsf(d1 - d2), 0.f);
    return fminf(d1, d2) - h * h * 0.25 / k;
}

D_FORCEINLINE float smoothIntersection(float d1, float d2, float k) { return -smoothUnion(-d1, -d2, k); }

D_FORCEINLINE float smoothDifference(float d1, float d2, float k) { return -smoothUnion(d1, -d2, k); }

struct CSGNode
{
    InstructionOp operation;
    uint32_t left;  // 4 bytes (bit 31 = leaf flag, bits 0-30 = left index)
    uint32_t right; // 4 bytes (bit 31 = leaf flag, bits 0-30 = right index)

    CSGNode() : operation(InstructionOp::Union), left(0), right(0) {}
    CSGNode(InstructionOp op, uint32_t leftIndex, uint32_t rightIndex)
        : operation(op), left(leftIndex), right(rightIndex)
    {
    }
    CSGNode(uint32_t leftIndex, uint32_t rightIndex)
        : operation(InstructionOp::Union), left(leftIndex), right(rightIndex)
    {
    }

    HD_FORCEINLINE bool leftIsLeaf() const { return (left & 0x80000000u) != 0; }

    HD_FORCEINLINE bool rightIsLeaf() const { return (right & 0x80000000u) != 0; }
    HD_FORCEINLINE uint32_t getLeftIndex() const { return left & 0x7FFFFFFFu; }
    HD_FORCEINLINE uint32_t getRightIndex() const { return right & 0x7FFFFFFFu; }
};

struct Instruction
{
    InstructionOp op;
    uint32_t operand; // Primitive index if op == Primitive, unused otherwise

    Instruction() : op(InstructionOp::Union), operand(0) {}
    Instruction(InstructionOp operation, uint32_t value = 0) : op(operation), operand(value) {}
};

struct CSGTree
{
    int materialIndex;
    PrimitiveData *primArray;
    float area = 1.f;

    // Compiled post-order instruction sequence
    Instruction *programArray;
    int programSize;

    __device__ float sdf(const float3 &point) const
    {
        // Evaluate pre-compiled post-order instruction sequence
        float stack[64];
        int sp = 0;

        for (int i = 0; i < programSize; ++i)
        {
            const Instruction &instr = programArray[i];

            switch (instr.op)
            {
            case InstructionOp::Primitive: {
                // Push primitive SDF value onto stack
                stack[sp++] = primArray[instr.operand].sdf(point);
                break;
            }
            case InstructionOp::Union: {
                // Pop 2, push min(a, b)
                float right = stack[--sp];
                float left = stack[--sp];
                stack[sp++] = fminf(left, right);
                break;
            }
            case InstructionOp::Intersection: {
                // Pop 2, push max(a, b)
                float right = stack[--sp];
                float left = stack[--sp];
                stack[sp++] = fmaxf(left, right);
                break;
            }
            case InstructionOp::Difference: {
                // Pop 2, push max(a, -b)
                float right = stack[--sp];
                float left = stack[--sp];
                stack[sp++] = fmaxf(left, -right);
                break;
            }
            case InstructionOp::SmoothUnion: {
                // Pop 2, push smooth union
                float right = stack[--sp];
                float left = stack[--sp];
                stack[sp++] = smoothUnion(left, right, 0.1f);
                break;
            }
            case InstructionOp::SmoothIntersection: {
                // Pop 2, push smooth intersection
                float right = stack[--sp];
                float left = stack[--sp];
                stack[sp++] = smoothIntersection(left, right, 0.1f);
                break;
            }
            case InstructionOp::SmoothDifference: {
                // Pop 2, push smooth difference
                float right = stack[--sp];
                float left = stack[--sp];
                stack[sp++] = smoothDifference(left, right, 0.1f);
                break;
            }
            }
        }

        return stack[0];
    }

    H_INLINE OptixAabb computeWorldAABB(const Matrix3x3 &rotation, const float3 &translation,
                                        const std::vector<PrimitiveData> &primitives,
                                        const std::vector<CSGNode> &nodesCPU) const
    {
        OptixAabb aabb;
        aabb.minX = 20000.f;
        aabb.minY = 20000.f;
        aabb.minZ = 20000.f;
        aabb.maxX = -20000.f;
        aabb.maxY = -20000.f;
        aabb.maxZ = -20000.f;
        for (int i = 0; i < nodesCPU.size(); ++i)
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
        aabb.minX = 20000.f;
        aabb.minY = 20000.f;
        aabb.minZ = 20000.f;
        aabb.maxX = -20000.f;
        aabb.maxY = -20000.f;
        aabb.maxZ = -20000.f;
        for (int i = 0; i < nodesCPU.size(); ++i)
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

    // Compile the tree into post-order instruction sequence
    void compileNode(uint32_t nodeIndex, std::vector<Instruction> &program, const std::vector<CSGNode> &p_nodes)
    {
        const CSGNode &node = p_nodes[nodeIndex];
        // printf("leftIsLeaf: %d, rightIsLeaf: %d, leftIndex: %d, rightIndex: %d\n", node.leftIsLeaf(),
        // node.rightIsLeaf(), node.getLeftIndex(), node.getRightIndex());
        //  Process left child
        if (node.leftIsLeaf())
        {
            program.push_back(Instruction(InstructionOp::Primitive, node.getLeftIndex()));
        }
        else
        {
            compileNode(node.getLeftIndex(), program, p_nodes);
        }

        // Process right child
        if (node.rightIsLeaf())
        {
            program.push_back(Instruction(InstructionOp::Primitive, node.getRightIndex()));
        }
        else
        {
            compileNode(node.getRightIndex(), program, p_nodes);
        }
        program.push_back(node.operation);
    }

    // Compile the entire tree (call this after constructing the tree)
    void compile(const std::vector<CSGNode> &nodes)
    {
        std::vector<Instruction> program;
        printf("Compiling CSGTree with %zu nodes\n", nodes.size());
        if (!nodes.empty())
        {
            compileNode(0, program, nodes);
        }
        programSize = program.size();
        printf("Compiled CSGTree with %d instructions\n", programSize);
        cudaMalloc(&programArray, programSize * sizeof(Instruction));
        cudaMemcpy(programArray, program.data(), programSize * sizeof(Instruction), cudaMemcpyHostToDevice);
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

    HD float getArea() const { return area; }

    uint32_t buildTree(uint32_t firstPrim, uint32_t lastPrim, std::vector<CSGNode> &nodes)
    {
        // Une seule primitive -> feuille
        if (firstPrim == lastPrim)
            return firstPrim | 0x80000000u;

        // Réserver le parent
        uint32_t nodeIndex = nodes.size();
        nodes.emplace_back();

        uint32_t mid = (firstPrim + lastPrim) / 2;

        uint32_t left = buildTree(firstPrim, mid, nodes);
        uint32_t right = buildTree(mid + 1, lastPrim, nodes);

        nodes[nodeIndex] = CSGNode(InstructionOp::Union, left, right);

        return nodeIndex;
    }
};