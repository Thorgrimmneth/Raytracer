#include "../lights/cuda_light.cuh"
#include "cuda_direct_lighting_integrator.cuh"

    __device__ __noinline__
    float3 DirectLightingIntegrator::directLighting( const CudaScene &	   p_scene,
													 const Ray &	   p_ray,
													 const HitRecord & p_hitRecord,
													 const float	   p_tMin,
													 const float	   p_tMax,
                                                     curandState* rng )
	{
		float3 Li = make_float3(0.0f);

		const Material& mtl = p_scene.materials[p_hitRecord.materialIndex];
		if(mtl.type == MaterialType::EMISSIVE)
		{
			return mtl.color * mtl.intensity;
		} 
		for (int i = 0; i < p_scene.nbLights; i++ )
		{
            const Light& light = p_scene.lights[i];
			if ( light.area>1e-6f )
			{
				
					float3 LiTemp = make_float3(0.0f);
					for ( int rayNumber = 0; rayNumber < nbSample; rayNumber++ )
					{
						LightSample lightSample = light.sample( p_hitRecord.point, rng);
						float angle = dot(p_hitRecord.normal, lightSample.direction);
						if(angle <= 0.f || lightSample.pdf <=0.f) continue;
						Ray			shadowRay	= Ray( p_hitRecord.point, lightSample.direction );
						shadowRay.offset( p_hitRecord.normal);
						if ( !p_scene.intersectAny( shadowRay, 1.e-4f, lightSample.distance ) )
						{
							float angle = max( dot( p_hitRecord.normal, lightSample.direction ), 0.f );
							float3 shade = mtl.getBSDF( p_ray, p_hitRecord, rng).brdf;
							// if the lightSample is invalid we add black (caused by cylinder light)
							if (length(lightSample.direction) < 1e-6f ) { LiTemp += make_float3(0.0f); }
							else {
								LiTemp += shade * lightSample.radiance * angle / lightSample.pdf; 
							}
						}
					}
					LiTemp /= nbSample;
					Li += LiTemp;
				
			}
			else
			{
				LightSample lightSample = light.sample( p_hitRecord.point );
				Ray			shadowRay	= Ray( p_hitRecord.point, lightSample.direction );
				shadowRay.offset( p_hitRecord.normal);
				if ( !p_scene.intersectAny( shadowRay, 0.f, lightSample.distance ) )
				{
					if (length(lightSample.direction) < 1e-6f ) { Li += make_float3(0.0f); }
					else{
					float angle = max( dot( p_hitRecord.normal, lightSample.direction ), 0.f );
					Li += mtl.getBSDF( p_ray, p_hitRecord, rng).brdf * lightSample.radiance * angle;
					}
				}
			}
		}
		return Li;
	}
