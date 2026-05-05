#pragma once

#include "../scene.cuh"
#include "../raytracingUtils/ray.cuh"
#include "../raytracingUtils/hitrecord.cuh"

struct DirectLightingIntegrator
{
	static const int nbSample = 4;

	__device__ __noinline__
	static float3 directLighting(const CudaScene &p_scene,
					   const Ray &p_ray,
					   const HitRecord &p_hitRecord,
					   const float p_tMin,
					   const float p_tMax,
					   curandState *rng);
};