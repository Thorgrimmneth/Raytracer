#pragma once

#include "../utils/cuda_op.cuh"

struct Ray{
    double time = 0;
	float3 origin;
	float3 direction;
	float3 invdir;

	__device__
	void offset( const float3 p_normal)
	{ 
			origin = pointAtT(time) + p_normal * 1.0e-3f ;
	}

	__device__
	float3 pointAtT(float t1) const{
		return origin + direction * t1;
	}

	__host__ __device__
    Ray() {}

    __host__ __device__
	Ray(const float3& o, const float3& d, float t = 0.f)
		: origin(o), direction(d), time(t)
	{
		invdir = make_float3(
			(fabsf(d.x) > 1e-8f) ? 1.f / d.x : 1e20f,
			(fabsf(d.y) > 1e-8f) ? 1.f / d.y : 1e20f,
			(fabsf(d.z) > 1e-8f) ? 1.f / d.z : 1e20f
		);
	}
};