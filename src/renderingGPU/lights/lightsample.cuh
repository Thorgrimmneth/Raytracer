#pragma once

#include "../utils/op.cuh"
struct LightSample
{
	float3 direction = make_float3(0.f);
	float distance = 1e20f;
	float3 radiance = make_float3(0.f,0.f,0.f);
	float pdf = 1.f;
	float power = 1.f;
};