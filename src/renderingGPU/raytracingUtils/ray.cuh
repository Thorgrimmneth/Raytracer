#pragma once

#include "../utils/op.cuh"
#include "../utils/macro.cuh"

struct Ray
{
    float time = 0.f;
    float3 origin;
    float3 direction;
    float3 invdir;

    D_FORCEINLINE 
	void offset(const float3 p_normal) { origin = pointAtT(time) + p_normal * 1.0e-3f; }

    D_FORCEINLINE 
	float3 pointAtT(float t1) const { return origin + direction * t1; }

    HD
	Ray() {}

    HD
	Ray(const float3 &o, const float3 &d, float t = 0.f) : origin(o), direction(d), time(t)
    {
        invdir = make_float3((fabsf(d.x) > 1e-8f) ? 1.f / d.x : 1e20f, (fabsf(d.y) > 1e-8f) ? 1.f / d.y : 1e20f,
                             (fabsf(d.z) > 1e-8f) ? 1.f / d.z : 1e20f);
    }
};