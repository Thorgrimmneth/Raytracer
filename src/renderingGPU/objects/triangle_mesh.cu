#include "triangle_mesh.cuh"


__device__ __noinline__
bool TriangleMesh::intersect(const Ray &p_ray,
                             const float p_tMin,
                             const float p_tMax,
                             HitRecord &p_hitRecord) const
{
    int stack[64];
    int stackPtr = 0;

    stack[stackPtr++] = 0; // root

    float tClosest = p_tMax;
    bool hit = false;

    while (stackPtr > 0)
    {
        int index = stack[--stackPtr];
        const BVH& node = bvhNodes[index];

        if (!node.bbox.intersect(p_ray, p_tMin, tClosest))
            continue;

        if (node.isLeaf())
        {
            for (int i = node.firstTriangleIndex; 
                 i < node.lastTriangleIndex; i++)
            {
                float t;
                float2 uv;

                if (triangles[i].intersect(p_ray, t, uv, vertices))
                {
                    if (t >= p_tMin && t < tClosest)
                    {
                        tClosest = t;
                        hit = true;

                        p_hitRecord.point = p_ray.pointAtT(t);
                        p_hitRecord.normal =
                            triangles[i].computeSmoothNormal(uv, normals);
                        p_hitRecord.faceNormal(p_ray.direction);
                        p_hitRecord.distance = t;
                        p_hitRecord.materialIndex = materialIndex;
                    }
                }
            }
        }
        else
        {
            stack[stackPtr++] = node.left;
            stack[stackPtr++] = node.right;
        }
    }

    return hit;
}


__device__ __noinline__
bool TriangleMesh::intersectAny(const Ray &p_ray,
                                const float p_tMin,
                                const float p_tMax,
                            const Material* materials) const
{
    if(materials[materialIndex].type() == MaterialType::TRANSPARENT) return false;
    int stack[64];
    int stackPtr = 0;

    stack[stackPtr++] = 0;

    while (stackPtr > 0)
    {
        int index = stack[--stackPtr];
        const BVH& node = bvhNodes[index];

        if (!node.bbox.intersect(p_ray, p_tMin, p_tMax))
            continue;

        if (node.isLeaf())
        {
            for (int i = node.firstTriangleIndex;
                 i < node.lastTriangleIndex; i++)
            {
                float t;
                float2 uv;

                if (triangles[i].intersect(p_ray, t, uv, vertices))
                {
                    if (t >= p_tMin && t < p_tMax)
                        return true;
                }
            }
        }
        else
        {
            stack[stackPtr++] = node.left;
            stack[stackPtr++] = node.right;
        }
    }

    return false;
}
