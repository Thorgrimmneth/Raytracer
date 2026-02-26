#pragma once

#include "../cuda_scene.cuh"
#include "../raytracingUtils/cuda_ray.cuh"
#include "../raytracingUtils/cuda_hitrecord.cuh"

struct DirectLightingIntegrator
{
	static const int nbSample = 64;
	float3 backgroundColor = float3f(0.f);

	__device__ __noinline__
	static float3 directLighting(const CudaScene &p_scene,
					   const Ray &p_ray,
					   const HitRecord &p_hitRecord,
					   const float p_tMin,
					   const float p_tMax,
					   curandState *rng);
};