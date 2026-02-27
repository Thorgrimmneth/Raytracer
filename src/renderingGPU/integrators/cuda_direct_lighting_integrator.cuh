#pragma once

#include "../cuda_scene.cuh"
#include "../raytracingUtils/cuda_ray.cuh"
#include "../raytracingUtils/cuda_hitrecord.cuh"

struct DirectLightingIntegrator
{
	static const int nbSample = 128;

	__device__ __noinline__
	static float3 directLighting(const CudaScene &p_scene,
					   const Ray &p_ray,
					   const HitRecord &p_hitRecord,
					   const float p_tMin,
					   const float p_tMax,
					   curandState *rng);
};