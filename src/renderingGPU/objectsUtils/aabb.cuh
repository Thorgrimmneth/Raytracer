#pragma once

#include "../utils/op.cuh"
#include "../raytracingUtils/ray.cuh"

struct AABB{
    float4 min = float4f(+INFINITY);
    float4 max = float4f(-INFINITY);

    __host__ __device__
    float4 centroid() const;

    __host__ __device__
    float area();

    __device__ __noinline__
    bool intersect( const Ray & ray,  float p_tMin,  float p_tMax ) const;

    __device__ __forceinline__
bool intersectCheck(
    const Ray& ray,
    float tMin,
    float tMax,
    float& outTMin) const
{
    for (int axis = 0; axis < 3; axis++)
    {
        float origin = (&ray.origin.x)[axis];
        float invDir = (&ray.invdir.x)[axis];
        float minVal = (&min.x)[axis];
        float maxVal = (&max.x)[axis];

        float t1 = (minVal - origin) * invDir;
        float t2 = (maxVal - origin) * invDir;

        float tNear = fminf(t1, t2);
        float tFar  = fmaxf(t1, t2);

        tMin = fmaxf(tMin, tNear);
        tMax = fminf(tMax, tFar);

        if (tMax < tMin)
            return false;
    }

    outTMin = tMin;
    return true;
}
    __host__
    void extend(const AABB& a);

    __host__
    void extend(const float3& a);

    __host__
    void extend(const float4& a);
};