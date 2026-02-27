#include "cuda_aabb.cuh"

__host__ __device__
float4 AABB::centroid() const
{
	return (min + max) * 0.5f;
}

__host__ __device__ 
float AABB::area()
{
	float3 minT = toFloat3(min);
	float3 maxT = toFloat3(max);
	float3 size = maxT - minT;
	return 2.0f * (size.x * size.y + size.x * size.z + size.y * size.z);
}

__device__ __noinline__
bool AABB::intersect(const Ray &ray, const float p_tMin, const float p_tMax) const
{
    float tmin = p_tMin;
    float tmax = p_tMax;

    // X
    float tx1 = (min.x - ray.origin.x) * ray.invdir.x;
    float tx2 = (max.x - ray.origin.x) * ray.invdir.x;

    tmin = fmaxf(tmin, fminf(tx1, tx2));
    tmax = fminf(tmax, fmaxf(tx1, tx2));

    // Y
    float ty1 = (min.y - ray.origin.y) * ray.invdir.y;
    float ty2 = (max.y - ray.origin.y) * ray.invdir.y;

    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));

    // Z
    float tz1 = (min.z - ray.origin.z) * ray.invdir.z;
    float tz2 = (max.z - ray.origin.z) * ray.invdir.z;

    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));

    return tmax >= tmin;
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