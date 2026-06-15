#pragma once

#include "../raytracingUtils/ray.cuh"
#include "../utils/macro.cuh"
#include "../utils/op.cuh"

struct AABB
{
    float3 min = make_float3(+INFINITY);
    float3 max = make_float3(-INFINITY);

    HOST 
    float3 centroid() const;

    HOST 
    float area() const;

    HOST 
    bool intersect(const Ray &ray, float tMin, float tMax, float &outTNear) const;

    D_FORCEINLINE bool intersectCheck(const float3 &origin, const float3 &direction, float tMin, float tMax, float &outTMin) const
    {
        float nearT = tMin;
        float farT = tMax;

        float3 t1 = (min - origin) / direction;
        float3 t2 = (max - origin) / direction;

        nearT = fmaxf(nearT, fmaxf(fminf(t1.x, t2.x), fmaxf(fminf(t1.y, t2.y), fminf(t1.z, t2.z))));
        farT = fminf(farT, fminf(fmaxf(t1.x, t2.x), fminf(fmaxf(t1.y, t2.y), fmaxf(t1.z, t2.z))));
        outTMin = nearT;
        return farT >= nearT;
        /*
        float t1 = (min.x - ray.origin.x) * 1.f / ray.direction.x;
        float t2 = (max.x - ray.origin.x) * 1.f / ray.direction.x;
        nearT = fmaxf(nearT, fminf(t1, t2));
        farT = fminf(farT, fmaxf(t1, t2));

        t1 = (min.y - ray.origin.y) * 1.f / ray.direction.y;
        t2 = (max.y - ray.origin.y) * 1.f / ray.direction.y;
        nearT = fmaxf(nearT, fminf(t1, t2));
        farT = fminf(farT, fmaxf(t1, t2));

        t1 = (min.z - ray.origin.z) * 1.f / ray.direction.z;
        t2 = (max.z - ray.origin.z) * 1.f / ray.direction.z;
        nearT = fmaxf(nearT, fminf(t1, t2));
        farT = fminf(farT, fmaxf(t1, t2));

        outTMin = nearT;
        return farT >= nearT;*/
    }

    D_FORCEINLINE
    bool intersectAnyGPU(const Ray &ray, float tMin, float tMax) const
    {
        const float tx1 = (min.x - ray.origin.x) * 1.f / ray.direction.x;
        const float tx2 = (max.x - ray.origin.x) * 1.f / ray.direction.x;

        const float ty1 = (min.y - ray.origin.y) * 1.f / ray.direction.y;
        const float ty2 = (max.y - ray.origin.y) * 1.f / ray.direction.y;

        const float tz1 = (min.z - ray.origin.z) * 1.f / ray.direction.z;
        const float tz2 = (max.z - ray.origin.z) * 1.f / ray.direction.z;

        const float txMin = fminf(tx1, tx2);
        const float txMax = fmaxf(tx1, tx2);

        const float tyMin = fminf(ty1, ty2);
        const float tyMax = fmaxf(ty1, ty2);

        const float tzMin = fminf(tz1, tz2);
        const float tzMax = fmaxf(tz1, tz2);

        const float nearT = fmaxf(tMin, fmaxf(txMin, fmaxf(tyMin, tzMin)));
        const float farT = fminf(tMax, fminf(txMax, fminf(tyMax, tzMax)));

        return farT >= nearT;
    }
    HOST 
    void extend(const AABB &a);

    HOST 
    void extend(const float3 &a);

    HOST 
    void extend(const float4 &a);

    HOST 
    bool isValid();
};