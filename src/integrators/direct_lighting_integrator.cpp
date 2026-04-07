#include "direct_lighting_integrator.hpp"
#include "lights/quad_light.hpp"
#include "materials/emissive_material.hpp"

namespace RT
{
	Vec3f DirectLightingIntegrator::Li( const Scene & p_scene,
										const Ray &	  p_ray,
										const float	  p_tMin,
										const float	  p_tMax ) const
	{
		HitRecord hitRecord;
		if ( p_scene.intersect( p_ray, p_tMin, p_tMax, hitRecord ) )
		{
			return _directLighting( p_scene, p_ray, hitRecord, p_tMin, p_tMax );
		}
		else {
			return _backgroundColor;
		}
	}

	Vec3f DirectLightingIntegrator::_directLighting( const Scene &	   p_scene,
													 const Ray &	   p_ray,
													 const HitRecord & p_hitRecord,
													 const float	   p_tMin,
													 const float	   p_tMax ) const
	{
		Vec3f Li = VEC3F_ZERO;

		BaseMaterial * mlt = p_hitRecord._object->getMaterial();

		// An emissive material acts like a light so we add it's emission color to Li
		if ( EmissiveMaterial * emissive = dynamic_cast<EmissiveMaterial *>( mlt ) )
		{
			if ( glm::dot( p_hitRecord._normal, -p_ray.getDirection() ) > 0.f ) { Li += emissive->getEmissionColor(); }
		}

		for ( const BaseLight * light : p_scene.getLights() )
		{
			if ( light->isSurface() )
			{
				Vec3f LiTemp = VEC3F_ZERO;
				for ( int rayNumber = 0; rayNumber < _nbLightSamples; rayNumber++ )
				{
					LightSample lightSample = light->sample( p_hitRecord._point );
					Ray			shadowRay	= Ray( p_hitRecord._point, lightSample._direction );
					shadowRay.offset( p_hitRecord._normal );
					if ( !p_scene.intersectAny( shadowRay, 1.e-4f, lightSample._distance ) )
					{
						float angle = glm::max( glm::dot( p_hitRecord._normal, lightSample._direction ), 0.f );
						Vec3f shade = mlt->shade( p_ray, p_hitRecord, lightSample );
						// if the lightSample is invalid we add black (caused by cylinder light)
						if ( lightSample._direction == VEC3F_ZERO ) { LiTemp += BLACK; }
						else { 
							LiTemp += shade * lightSample._radiance * angle; 
						}
					}
				}
				LiTemp /= _nbLightSamples;
				Li += LiTemp;
			}
			else
			{
				LightSample lightSample = light->sample( p_hitRecord._point );
				Ray			shadowRay	= Ray( p_hitRecord._point, lightSample._direction );
				shadowRay.offset( p_hitRecord._normal );
				if ( !p_scene.intersectAny( shadowRay, 0.f, lightSample._distance ) )
				{
					float angle = glm::max( glm::dot( p_hitRecord._normal, lightSample._direction ), 0.f );
					Li += mlt->shade( p_ray, p_hitRecord, lightSample ) * lightSample._radiance * angle;
				}
			}
		}

		return Li + (_backgroundColor * 0.3f * mlt->getFlatColor()) ;
	}

} // namespace RT