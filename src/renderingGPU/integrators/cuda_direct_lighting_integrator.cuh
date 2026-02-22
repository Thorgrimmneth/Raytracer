#pragma once

#include "../cuda_scene.cuh"
#include "../raytracingUtils/cuda_ray.cuh"
#include "../raytracingUtils/cuda_hitrecord.cuh"

struct DirectLightingIntegrator
{
	int nbSample = 128;
	float3 backgroundColor = float3f(0.f);

	__device__
	float3 directLighting(const CudaScene &p_scene,
					   const Ray &p_ray,
					   const HitRecord &p_hitRecord,
					   const float p_tMin,
					   const float p_tMax,
					   curandState *rng) const;

	__device__
	float3 LI(const CudaScene &p_scene, const Ray &p_ray, const float p_tMin, const float p_tMax, curandState *rng);
};