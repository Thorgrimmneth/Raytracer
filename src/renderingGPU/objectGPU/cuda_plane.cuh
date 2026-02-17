#pragma once

#include "cuda_op.cuh"
#include "cuda_ray.cuh"
#include "cuda_hitrecord.cuh"

struct Plane
{
	float4 normal;
    float delta;
    int materialIndex;

	__device__
    bool intersectGeometry( const Ray & ray, float & p_t1 ) const;

	__device__
	bool intersect( const Ray & p_ray, const float p_tMin, const float p_tMax, HitRecord & p_hitRecord ) const;

	__device__
	bool intersectAny( const Ray & p_ray, const float p_tMin, const float p_tMax ) const;
};