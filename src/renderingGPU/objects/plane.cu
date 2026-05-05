#include "plane.cuh"
#include <stdio.h>

__device__
bool Plane::intersectGeometry(const Ray &ray, float &t) const
{
    float ND = dot(normal, ray.direction);
    if (fabsf(ND) < 1e-6f)
        return false;

    t = -(dot(normal, ray.origin) + delta) / ND;
    return t >= 0.f;
}

__device__
bool Plane::intersect(const Ray &p_ray, const float p_tMin, const float p_tMax, HitRecord &p_hitRecord) const
{
	float t1;
	if (intersectGeometry(p_ray, t1))
	{
		if (t1 < p_tMin || t1 > p_tMax)
		{
			return false;
		} // not in range

		// Intersection found, fill p_hitRecord.
		p_hitRecord.point = p_ray.pointAtT(t1);
		p_hitRecord.normal = normal;
		p_hitRecord.faceNormal(p_ray.direction);
		p_hitRecord.distance = t1;
		p_hitRecord.materialIndex = materialIndex;

		return true;
	}
	return false;
}

__device__
bool Plane::intersectAny(const Ray &p_ray, const float p_tMin, const float p_tMax, const Material* materials) const
{
	if(materials[materialIndex].type == MaterialType::TRANSPARENT){
		return false;
	}
	float t1;
	if (intersectGeometry(p_ray, t1))
	{
		if (t1 < p_tMin || t1 > p_tMax)
		{
			return false;
		} // not in range
		return true;
	}
	return false;
}