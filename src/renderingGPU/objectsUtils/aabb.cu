#include "aabb.cuh"

HOST
float3 AABB::centroid() const
{
	return (min + max) * 0.5f;
}

HOST
float AABB::area() const
{
	float3 size = max - min;
	return 2.0f * (size.x * size.y + size.x * size.z + size.y * size.z);
}

HOST
void AABB::extend(const AABB& a){
    min = getMin(min, a.min);
    max = getMax(max, a.max);
}

HOST
void AABB::extend(const float3& a){
    min = getMin(min, a);
    max = getMax(max, a);
}

HOST
void AABB::extend(const float4& a){
    extend(make_float3(a));
}

HOST
bool AABB::isValid()
{
    return min.x <= max.x && min.y <= max.y && min.z <= max.z;
}