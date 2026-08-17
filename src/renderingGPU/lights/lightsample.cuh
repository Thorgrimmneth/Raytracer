#pragma once

struct LightSample
{
	float3 direction;
	float distance;
	float3 radiance;
	float pdf;
	float power;
};