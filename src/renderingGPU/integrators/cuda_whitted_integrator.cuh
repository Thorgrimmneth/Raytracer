#pragma once

#include "../cuda_scene.cuh"
#include "../raytracingUtils/cuda_ray.cuh"
#include "../raytracingUtils/cuda_hitrecord.cuh"
#include "cuda_direct_lighting_integrator.cuh"

struct WhittedIntegrator{
    int nbBounces = 5;
    float3		sunDirection				  = sunDirectionFromAngles(50.f, 20.f);
	int			skyColorSamples			  = 32;
    const float hr = 8.0f;
	const float hm			 = 1.2f;
	const float3 betaR		 = make_float3( 3.8e-6f, 13.5e-6f, 33.1e-6f ) * 1000.f;
	const float3 betaM		 = float3f( 21e-6f ) * 1000.f;
	const float	exposure					  = 20.f;

    __device__
    float3 sunDirectionFromAngles( float elevation, float azimuth );

    __device__
    float3 lighting(
        const CudaScene& scene,
        const Ray& primaryRay,
        const float tMin,
        const float tMax,
        curandState* rng
    ) const;

    __device__
    float3 toneMap( const float3 & c ) const;

    __device__
    float3 getSkyColor( const Ray & p_ray ) const;

    __device__
    float3 Li( const CudaScene & p_scene,
								 const Ray &   p_ray,
								 const float   p_tMin,
								 const float   p_tMax,
                                 curandState* rng ) const;
};