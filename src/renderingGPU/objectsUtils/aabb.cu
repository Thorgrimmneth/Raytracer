#include "aabb.cuh"

HOST
float4 AABB::centroid() const
{
	return (min + max) * 0.5f;
}

HOST
float AABB::area() const
{
	float3 minT = toFloat3(min);
	float3 maxT = toFloat3(max);
	float3 size = maxT - minT;
	return 2.0f * (size.x * size.y + size.x * size.z + size.y * size.z);
}

HOST
void AABB::extend(const AABB& a){
    min = getMin(min, a.min);
    max = getMax(max, a.max);
}

HOST
void AABB::extend(const float3& a){
    min = getMin(min, toFloat4(a));
    max = getMax(max, toFloat4(a));
}

HOST
void AABB::extend(const float4& a){
    min = getMin(min, a);
    max = getMax(max, a);
}

HOST
bool AABB::isValid()
{
    return min.x <= max.x && min.y <= max.y && min.z <= max.z;
}