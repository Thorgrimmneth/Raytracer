#include "whitted_integrator.hpp"
#include "utils/random.hpp"

namespace RT
{
	Vec3f WhittedIntegrator::Li( const Scene & p_scene,
								 const Ray &   p_ray,
								 const float   p_tMin,
								 const float   p_tMax ) const
	{
		HitRecord hitRecord;
		return _recursiveLighting( p_scene, p_ray, hitRecord, p_tMin, p_tMax, 0, false );
	}

	Vec3f WhittedIntegrator::_recursiveLighting( const Scene & p_scene,
												 const Ray &   p_ray,
												 HitRecord &   p_hitRecord,
												 const float   p_tMin,
												 const float   p_tMax,
												 int		   bounce,
												 bool		   isInMaterial ) const
	{
		if ( bounce >= _nbBounces ) { return BLACK; }

		if ( p_scene.intersect( p_ray, p_tMin, p_tMax, p_hitRecord ) )
		{
			// It the ray hits a mirror we need to reflect the ray
			if ( p_hitRecord._object->getMaterial()->isMirror() )
			{
				Vec3f direction = p_ray.getDirection();
				Vec3f normal	= p_hitRecord._normal;

				Vec3f reflectedRayDir = glm::reflect( direction, normal );
				Ray	  reflectedRay	  = Ray( p_hitRecord._point, reflectedRayDir );
				reflectedRay.offset( p_hitRecord._normal );

				return _recursiveLighting(
					p_scene, reflectedRay, p_hitRecord, p_tMin, p_tMax, bounce + 1, isInMaterial );
			}
			// It the ray hits a transparent object we need to reflect and refract the ray
			else if ( p_hitRecord._object->getMaterial()->isTransparent() )
			{
				float n1	 = 1.f; // 1.f in void/air
				float n2	 = p_hitRecord._object->getMaterial()->getIOR();
				Vec3f normal = p_hitRecord._normal;
				if ( isInMaterial )
				{
					std::swap( n1, n2 ); // swap n1 and n2 if we are inside the material
				}

				Vec3f rayDir = p_ray.getDirection();

				float cosI = glm::dot( normal, -rayDir );

				float sinT = ( n1 / n2 ) * sqrt( 1.0f - cosI * cosI );

				// total reflection
				if ( sinT > 1.0f )
				{
					Vec3f reflectedRayDir = glm::reflect( rayDir, normal );
					Ray	  reflectedRay( p_hitRecord._point, reflectedRayDir );
					reflectedRay.offset( p_hitRecord._normal );
					HitRecord reflectHitRecord;
					return _recursiveLighting(
						p_scene, reflectedRay, reflectHitRecord, p_tMin, p_tMax, bounce + 1, isInMaterial );
				}

				// Wmp the sinT and cosI to avoid NaN
				sinT	   = glm::clamp( sinT, -1.0f, 1.0f );
				cosI	   = glm::clamp( cosI, -1.0f, 1.0f );
				float cosT = sqrt( 1 - sinT * sinT );
				cosT	   = glm::clamp( cosT, -1.0f, 1.0f );

				float rs = ( ( n1 * cosI ) - ( n2 * cosT ) ) / ( ( n1 * cosI ) + ( n2 * cosT ) );
				rs		 = rs * rs;
				float rp = ( ( n1 * cosT ) - ( n2 * cosI ) ) / ( ( n1 * cosT ) + ( n2 * cosI ) );
				rp		 = rp * rp;

				float reff = ( rs + rp ) / 2.f;

				Vec3f reflectedRayDir = glm::reflect( rayDir, normal );
				Ray	  reflectedRay	  = Ray( p_hitRecord._point, reflectedRayDir );
				reflectedRay.offset( normal );

				HitRecord reflectHitRecord;
				Vec3f	  reflectedColor = _recursiveLighting(
					p_scene, reflectedRay, reflectHitRecord, p_tMin, p_tMax, bounce + 1, isInMaterial );

				Vec3f refractedRayDir = glm::refract( rayDir, normal, n1 / n2 );
				Ray	  refractedRay	  = Ray( p_hitRecord._point, refractedRayDir );
				refractedRay.offset( -normal ); // ray in the sphere so the normal is inverted

				HitRecord refractHitRecord;
				Vec3f	  refractedColor = _recursiveLighting(
					p_scene, refractedRay, refractHitRecord, p_tMin, p_tMax, bounce + 1, !isInMaterial );

				return ( 1.f - reff ) * refractedColor + reff * reflectedColor;
			}
			// If the ray hits a diffuse object we just need to compute the direct lighting
			else { return DirectLightingIntegrator::_directLighting( p_scene, p_ray, p_hitRecord, p_tMin, p_tMax ); }
		}


		float tMin = 0.f;
		float tMax = 1e10f;
		Vec3f skyColor = _getSkyColor(p_ray);
		//std::cout << skyColor.x << "," << skyColor.y << "," << skyColor.z << "\n";
		skyColor *= _exposure;
		skyColor	   = _toneMap( skyColor);
		return skyColor;
	}

	Vec3f WhittedIntegrator::_toneMap( const Vec3f & c ) const{
		return c / ( Vec3f( 1.f ) + c );
	}

	Vec3f WhittedIntegrator::_getSkyColor( const Ray & p_ray ) const
	{
		Vec3f rayDir = normalize( p_ray.getDirection() );
		Vec3f sunDir = normalize( _sunDirection );

		float tMax = 100.f;
		float dt   = tMax / _skyColorSamples;

		Vec3f sumR = VEC3F_ZERO;
		Vec3f sumM = VEC3F_ZERO;

		float opticalDepthR = 0.f;
		float opticalDepthM = 0.f;

		float mu = dot( rayDir, sunDir );

		float phaseR = ( 3.f / ( 16.f * PIf ) ) * ( 1.f + mu * mu );

		float g		 = 0.76f;
		float phaseM = ( 3.f / ( 8.f * PIf ) ) * ( ( 1.f - g * g ) * ( 1.f + mu * mu ) )
					   / ( ( 2.f + g * g ) * pow( 1.f + g * g - 2.f * g * mu, 1.5f ) );

		float sunBelow = std::max( 0.f, -sunDir.y );

		for ( int i = 0; i < _skyColorSamples; ++i )
		{
			float t = ( i + 0.5f ) * dt;
			Vec3f p = p_ray.pointAtT( t );

			float height = std::max( 0.f, p.y );

			float altitudeFade = exp( -height / 5.f );
			float horizonFade  = exp( -sunBelow * 20.f * altitudeFade );

			float hr = exp( -height / _hr ) * dt * horizonFade;
			float hm = exp( -height / _hm ) * dt * horizonFade;

			opticalDepthR += hr;
			opticalDepthM += hm;

			float sunOpticalDepthR = 0.f;
			float sunOpticalDepthM = 0.f;

			int	  sunSamples = 8;
			float sunDt		 = 100.f / sunSamples;

			for ( int j = 0; j < sunSamples; ++j )
			{
				float ts = ( j + 0.5f ) * sunDt;
				Vec3f ps = p + sunDir * ts;

				float h = std::max( 0.f, ps.y );

				sunOpticalDepthR += exp( -h / _hr ) * sunDt;
				sunOpticalDepthM += exp( -h / _hm ) * sunDt;
			}

			Vec3f tau = _betaR * ( opticalDepthR + sunOpticalDepthR ) + _betaM * ( opticalDepthM + sunOpticalDepthM );

			Vec3f attenuation( exp( -tau.x ), exp( -tau.y ), exp( -tau.z ) );

			sumR += attenuation * hr;
			sumM += attenuation * hm;
		}

		return ( sumR * _betaR * phaseR + sumM * _betaM * phaseM );
	}



} // namespace RT