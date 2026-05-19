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


__host__ __device__
bool AABB::intersect(
    const Ray& ray,
    float tMin,
    float tMax,
    float& outTNear,
    float& outTFar) const
{
    outTNear = tMin;
    outTFar = tMax;

    for (int axis = 0; axis < 3; ++axis)
    {
        float origin =
            axis == 0 ? ray.origin.x :
            axis == 1 ? ray.origin.y :
                        ray.origin.z;

        float direction =
            axis == 0 ? ray.direction.x :
            axis == 1 ? ray.direction.y :
                        ray.direction.z;

        float minA =
            axis == 0 ? min.x :
            axis == 1 ? min.y :
                        min.z;

        float maxA =
            axis == 0 ? max.x :
            axis == 1 ? max.y :
                        max.z;

        float invD = 1.0f / direction;

        float t0 = (minA - origin) * invD;
        float t1 = (maxA - origin) * invD;

        if (invD < 0.0f)
        {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        outTNear = fmaxf(outTNear, t0);
        outTFar = fminf(outTFar, t1);

        if (outTFar <= outTNear)
            return false;
    }

    return true;
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