#include "triangle_mesh.cuh"


__device__ __noinline__
bool TriangleMesh::intersect(
    const Ray& p_ray,
    const float p_tMin,
    const float p_tMax,
    HitRecord& p_hitRecord) const
{
    constexpr int STACK_SIZE = 256;

    if (bvhNodes == nullptr ||
        triangles == nullptr ||
        vertices == nullptr ||
        normals == nullptr)
    {
        return false;
    }

    if (bvhNodeCount <= 0 ||
        triangleCount <= 0 ||
        vertexCount <= 0)
    {
        return false;
    }

    int stack[STACK_SIZE];
    int stackPtr = 0;

    stack[stackPtr++] = 0;

    float tClosest = p_tMax;
    bool hit = false;

    while (stackPtr > 0)
    {
        int nodeIndex = stack[--stackPtr];

        if (nodeIndex < 0 || nodeIndex >= bvhNodeCount)
        {
            return hit;
        }

        const BVH& node = bvhNodes[nodeIndex];

        if (!node.bbox.intersect(p_ray, p_tMin, tClosest))
            continue;

        if (node.isLeaf())
        {
            int first = node.firstTriangleIndex;
            int last = node.lastTriangleIndex;

            if (first < 0 || first >= triangleCount)
                continue;

            if (last < 0 || last >= triangleCount)
                continue;

            if (last < first)
                continue;

            for (int i = first; i <= last; ++i)
            {
                const TriangleMeshGeometry& tri = triangles[i];

                if (tri.i0 < 0 || tri.i0 >= vertexCount ||
                    tri.i1 < 0 || tri.i1 >= vertexCount ||
                    tri.i2 < 0 || tri.i2 >= vertexCount)
                {
                    continue;
                }

                float t;
                float2 uv;

                if (tri.intersect(p_ray, t, uv, vertices))
                {
                    if (t >= p_tMin && t < tClosest)
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
        }
        else
        {
            if (node.left < 0 || node.left >= bvhNodeCount)
                continue;

            if (node.right < 0 || node.right >= bvhNodeCount)
                continue;

            if (stackPtr + 2 > STACK_SIZE)
            {
                return hit;
            }

            stack[stackPtr++] = node.left;
            stack[stackPtr++] = node.right;
        }
    }

    return hit;
}


__device__ __noinline__
bool TriangleMesh::intersectAny(
    const Ray& p_ray,
    const float p_tMin,
    const float p_tMax,
    const Material* materials) const
{
    constexpr int STACK_SIZE = 256;

    if (bvhNodes == nullptr ||
        triangles == nullptr ||
        vertices == nullptr)
    {
        return false;
    }

    if (bvhNodeCount <= 0 ||
        triangleCount <= 0 ||
        vertexCount <= 0)
    {
        return false;
    }

    // Temporarily comment this out unless you can prove materialIndex is valid.
    // if (materials[materialIndex].type() == MaterialType::TRANSPARENT)
    //     return false;

    int stack[STACK_SIZE];
    int stackPtr = 0;

    stack[stackPtr++] = 0;

    while (stackPtr > 0)
    {
        int nodeIndex = stack[--stackPtr];

        if (nodeIndex < 0 || nodeIndex >= bvhNodeCount)
            return false;

        const BVH& node = bvhNodes[nodeIndex];

        if (!node.bbox.intersect(p_ray, p_tMin, p_tMax))
            continue;

        if (node.isLeaf())
        {
            int first = node.firstTriangleIndex;
            int last = node.lastTriangleIndex;

            if (first < 0 || first >= triangleCount)
                continue;

            if (last < 0 || last >= triangleCount)
                continue;

            if (last < first)
                continue;

            for (int i = first; i <= last; ++i)
            {
                const TriangleMeshGeometry& tri = triangles[i];

                if (tri.i0 < 0 || tri.i0 >= vertexCount ||
                    tri.i1 < 0 || tri.i1 >= vertexCount ||
                    tri.i2 < 0 || tri.i2 >= vertexCount)
                {
                    continue;
                }

                float t;
                float2 uv;

                if (tri.intersect(p_ray, t, uv, vertices))
                {
                    if (t >= p_tMin && t < p_tMax)
                        return true;
                }
            }
        }
        else
        {
            if (node.left < 0 || node.left >= bvhNodeCount)
                continue;

            if (node.right < 0 || node.right >= bvhNodeCount)
                continue;

            if (stackPtr + 2 > STACK_SIZE)
                return false;

            stack[stackPtr++] = node.left;
            stack[stackPtr++] = node.right;
        }
    }

    return false;
}
