#include "triangle_mesh.cuh"


__device__
bool TriangleMesh::intersect(
    const Ray& p_ray,
    const float p_tMin,
    const float p_tMax,
    HitRecord& p_hitRecord) const
{
    constexpr int STACK_SIZE = 128;

#ifdef DEBUG_BVH
    if (bvhNodes == nullptr ||
        triangleRefIndices == nullptr ||
        triangles == nullptr ||
        vertices == nullptr ||
        normals == nullptr)
    {
        return false;
    }

    if (bvhNodeCount <= 0 ||
        triangleCount <= 0 ||
        refCount <= 0 ||
        vertexCount <= 0)
    {
        return false;
    }
#endif

    int stack[STACK_SIZE];
    int stackPtr = 0;

    stack[stackPtr++] = 0;

    float tClosest = p_tMax;
    bool hit = false;

    while (stackPtr > 0)
    {
        const int nodeIndex = stack[--stackPtr];

#ifdef DEBUG_BVH
        if ((unsigned)nodeIndex >= (unsigned)bvhNodeCount)
            continue;
#endif

        const BVH& node = bvhNodes[nodeIndex];

        float nodeTNear;

        if (!node.bbox.intersectCheck(
                p_ray,
                p_tMin,
                tClosest,
                nodeTNear))
        {
            continue;
        }

        if (node.isLeaf())
        {
            const int first = node.firstRefIndex;
            const int count = node.refCount;

#ifdef DEBUG_BVH
            if (first < 0 || count <= 0)
                continue;

            if (first + count > refCount)
                continue;
#endif

            for (int localIdx = 0; localIdx < count; ++localIdx)
            {
                const int refIndex = first + localIdx;

#ifdef DEBUG_BVH
                if ((unsigned)refIndex >= (unsigned)refCount)
                    continue;
#endif

                const int triIndex = triangleRefIndices[refIndex];

#ifdef DEBUG_BVH
                if ((unsigned)triIndex >= (unsigned)triangleCount)
                    continue;
#endif

                const TriangleMeshGeometry& tri = triangles[triIndex];

#ifdef DEBUG_BVH
                if ((unsigned)tri.i0 >= (unsigned)vertexCount ||
                    (unsigned)tri.i1 >= (unsigned)vertexCount ||
                    (unsigned)tri.i2 >= (unsigned)vertexCount)
                {
                    continue;
                }
#endif

                float t;
                float2 uv;

                if (tri.intersect(
                        p_ray,
                        p_tMin,
                        tClosest,
                        t,
                        uv))
                {
                    tClosest = t;
                    hit = true;

                    p_hitRecord.point = p_ray.pointAtT(t);
                    p_hitRecord.normal = tri.computeSmoothNormal(uv, normals);
                    p_hitRecord.faceNormal(p_ray.direction);
                    p_hitRecord.distance = t;
                    p_hitRecord.materialIndex = materialIndex;
                }
            }
        }
        else
        {
            const int left = node.left;
            const int right = node.right;

#ifdef DEBUG_BVH
            if ((unsigned)left >= (unsigned)bvhNodeCount ||
                (unsigned)right >= (unsigned)bvhNodeCount)
            {
                continue;
            }
#endif

            float leftTNear;
            float rightTNear;

            const bool hitLeft = bvhNodes[left].bbox.intersectCheck(
                p_ray,
                p_tMin,
                tClosest,
                leftTNear
            );

            const bool hitRight = bvhNodes[right].bbox.intersectCheck(
                p_ray,
                p_tMin,
                tClosest,
                rightTNear
            );

            if (hitLeft && hitRight)
            {
                if (stackPtr + 2 > STACK_SIZE)
                    return hit;

                if (leftTNear < rightTNear)
                {
                    stack[stackPtr++] = right;
                    stack[stackPtr++] = left;
                }
                else
                {
                    stack[stackPtr++] = left;
                    stack[stackPtr++] = right;
                }
            }
            else if (hitLeft)
            {
                if (stackPtr + 1 > STACK_SIZE)
                    return hit;

                stack[stackPtr++] = left;
            }
            else if (hitRight)
            {
                if (stackPtr + 1 > STACK_SIZE)
                    return hit;

                stack[stackPtr++] = right;
            }
        }
    }

    return hit;
}


__device__
bool TriangleMesh::intersectAny(
    const Ray& p_ray,
    const float p_tMin,
    const float p_tMax,
    const Material* materials) const
{
    constexpr int STACK_SIZE = 128;

#ifdef DEBUG_BVH
    if (bvhNodes == nullptr ||
        triangleRefIndices == nullptr ||
        triangles == nullptr ||
        vertices == nullptr)
    {
        return false;
    }

    if (bvhNodeCount <= 0 ||
        triangleCount <= 0 ||
        refCount <= 0 ||
        vertexCount <= 0)
    {
        return false;
    }
#endif

    // Si tu veux gérer les matériaux transparents plus tard,
    // il faudra probablement tester par triangle / matériau réel.
    // Pour l'instant, on ne skip pas le mesh entier ici.

    int stack[STACK_SIZE];
    int stackPtr = 0;

    stack[stackPtr++] = 0;

    while (stackPtr > 0)
    {
        const int nodeIndex = stack[--stackPtr];

#ifdef DEBUG_BVH
        if ((unsigned)nodeIndex >= (unsigned)bvhNodeCount)
            continue;
#endif

        const BVH& node = bvhNodes[nodeIndex];

        float nodeTNear;

        if (!node.bbox.intersectCheck(
                p_ray,
                p_tMin,
                p_tMax,
                nodeTNear))
        {
            continue;
        }

        if (node.isLeaf())
        {
            const int first = node.firstRefIndex;
            const int count = node.refCount;

#ifdef DEBUG_BVH
            if (first < 0 || count <= 0)
                continue;

            if (first + count > refCount)
                continue;
#endif

            for (int localIdx = 0; localIdx < count; ++localIdx)
            {
                const int refIndex = first + localIdx;

#ifdef DEBUG_BVH
                if ((unsigned)refIndex >= (unsigned)refCount)
                    continue;
#endif

                const int triIndex = triangleRefIndices[refIndex];

#ifdef DEBUG_BVH
                if ((unsigned)triIndex >= (unsigned)triangleCount)
                    continue;
#endif

                const TriangleMeshGeometry& tri = triangles[triIndex];

#ifdef DEBUG_BVH
                if ((unsigned)tri.i0 >= (unsigned)vertexCount ||
                    (unsigned)tri.i1 >= (unsigned)vertexCount ||
                    (unsigned)tri.i2 >= (unsigned)vertexCount)
                {
                    continue;
                }
#endif

                float t;
                float2 uv;

                if (tri.intersect(
                        p_ray,
                        p_tMin,
                        p_tMax,
                        t,
                        uv))
                {
                    return true;
                }
            }
        }
        else
        {
            const int left = node.left;
            const int right = node.right;

#ifdef DEBUG_BVH
            if ((unsigned)left >= (unsigned)bvhNodeCount ||
                (unsigned)right >= (unsigned)bvhNodeCount)
            {
                continue;
            }
#endif

            float leftTNear;
            float rightTNear;

            const bool hitLeft = bvhNodes[left].bbox.intersectCheck(
                p_ray,
                p_tMin,
                p_tMax,
                leftTNear
            );

            const bool hitRight = bvhNodes[right].bbox.intersectCheck(
                p_ray,
                p_tMin,
                p_tMax,
                rightTNear
            );

            if (hitLeft && hitRight)
            {
                if (stackPtr + 2 > STACK_SIZE)
                    return false;

                // Stack LIFO :
                // on push le plus loin d'abord pour visiter le plus proche en premier.
                if (leftTNear < rightTNear)
                {
                    stack[stackPtr++] = right;
                    stack[stackPtr++] = left;
                }
                else
                {
                    stack[stackPtr++] = left;
                    stack[stackPtr++] = right;
                }
            }
            else if (hitLeft)
            {
                if (stackPtr + 1 > STACK_SIZE)
                    return false;

                stack[stackPtr++] = left;
            }
            else if (hitRight)
            {
                if (stackPtr + 1 > STACK_SIZE)
                    return false;

                stack[stackPtr++] = right;
            }
        }
    }

    return false;
}
