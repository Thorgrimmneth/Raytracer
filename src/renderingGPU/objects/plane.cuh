#pragma once

#include "../utils/op.cuh"
#include "../raytracingUtils/ray.cuh"
#include "../raytracingUtils/hitrecord.cuh"
#include "../materials/material.cuh"

struct Plane
{
	float3 normal;
    float delta;
    int materialIndex;

	Plane() = default;
	Plane(float3 pos, float3 n) : normal(n), delta(dot(-n, pos)) {}
	__device__
    bool intersectGeometry( const Ray & ray, float & p_t1 ) const;

	__device__
	bool intersect( const Ray & p_ray, const float p_tMin, const float p_tMax, HitRecord & p_hitRecord ) const;

	__device__
	bool intersectAny( const Ray & p_ray, const float p_tMin, const float p_tMax, const Material* materials ) const;
};