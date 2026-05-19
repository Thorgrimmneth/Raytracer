#include "aabb.cuh"

__host__ __device__
float4 AABB::centroid() const
{
	return (min + max) * 0.5f;
}

__host__ __device__ 
float AABB::area() const
{
	float3 minT = toFloat3(min);
	float3 maxT = toFloat3(max);
	float3 size = maxT - minT;
	return 2.0f * (size.x * size.y + size.x * size.z + size.y * size.z);
}


__host__
bool AABB::intersect(
    const Ray& ray,
    float tMin,
    float tMax,
    float& outTNear) const
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

    outTNear = nearT;

    return farT >= nearT;
}
    

__host__
void AABB::extend(const AABB& a){
    min = getMin(min, a.min);
    max = getMax(max, a.max);
}

__host__
void AABB::extend(const float3& a){
    min = getMin(min, toFloat4(a));
    max = getMax(max, toFloat4(a));
}

__host__
void AABB::extend(const float4& a){
    min = getMin(min, a);
    max = getMax(max, a);
}

__host__
bool AABB::isValid()
{
    return min.x <= max.x && min.y <= max.y && min.z <= max.z;
}