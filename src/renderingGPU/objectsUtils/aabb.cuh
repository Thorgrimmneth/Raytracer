#pragma once

#include "../utils/op.cuh"
#include "../raytracingUtils/ray.cuh"

struct AABB{
    float4 min = float4f(+INFINITY);
    float4 max = float4f(-INFINITY);

    __host__ __device__
    float4 centroid() const;

    __host__ __device__
    float area() const;

    __host__
    bool intersect( const Ray& ray,
    float tMin,
    float tMax,
    float& outTNear) const;

__device__ __forceinline__
bool intersectCheck(
    const Ray& ray,
    float tMin,
    float tMax,
    float& outTMin) const
{
    float nearT = tMin;
    float farT  = tMax;

    float t1 = (min.x - ray.origin.x) * ray.invdir.x;
    float t2 = (max.x - ray.origin.x) * ray.invdir.x;
    nearT = fmaxf(nearT, fminf(t1, t2));
    farT  = fminf(farT,  fmaxf(t1, t2));

    t1 = (min.y - ray.origin.y) * ray.invdir.y;
    t2 = (max.y - ray.origin.y) * ray.invdir.y;
    nearT = fmaxf(nearT, fminf(t1, t2));
    farT  = fminf(farT,  fmaxf(t1, t2));

    t1 = (min.z - ray.origin.z) * ray.invdir.z;
    t2 = (max.z - ray.origin.z) * ray.invdir.z;
    nearT = fmaxf(nearT, fminf(t1, t2));
    farT  = fminf(farT,  fmaxf(t1, t2));

    outTMin = nearT;
    return farT >= nearT;
}

    __device__ __forceinline__
bool intersectAnyGPU(
    const Ray& ray,
    float tMin,
    float tMax) const
{
    const float tx1 = (min.x - ray.origin.x) * ray.invdir.x;
    const float tx2 = (max.x - ray.origin.x) * ray.invdir.x;

    const float ty1 = (min.y - ray.origin.y) * ray.invdir.y;
    const float ty2 = (max.y - ray.origin.y) * ray.invdir.y;

    const float tz1 = (min.z - ray.origin.z) * ray.invdir.z;
    const float tz2 = (max.z - ray.origin.z) * ray.invdir.z;

    const float txMin = fminf(tx1, tx2);
    const float txMax = fmaxf(tx1, tx2);

    const float tyMin = fminf(ty1, ty2);
    const float tyMax = fmaxf(ty1, ty2);

    const float tzMin = fminf(tz1, tz2);
    const float tzMax = fmaxf(tz1, tz2);

    const float nearT = fmaxf(tMin, fmaxf(txMin, fmaxf(tyMin, tzMin)));
    const float farT  = fminf(tMax, fminf(txMax, fminf(tyMax, tzMax)));

    return farT >= nearT;
}
    __host__
    void extend(const AABB& a);

    __host__
    void extend(const float3& a);

    __host__
    void extend(const float4& a);

    __host__
    bool isValid();
};