#ifndef __RT_ISICG_WHITTED_INTEGRATOR__
#define __RT_ISICG_WHITTED_INTEGRATOR__

#include "direct_lighting_integrator.hpp"

namespace RT
{
	class WhittedIntegrator : public DirectLightingIntegrator
	{
	  public:
		WhittedIntegrator() : DirectLightingIntegrator() {}

		virtual ~WhittedIntegrator() = default;

		const IntegratorType getType() const override { return IntegratorType::DIRECT_LIGHTING; }

		// Return incoming luminance.
		Vec3f Li( const Scene & p_scene, const Ray & p_ray, const float p_tMin, const float p_tMax ) const override;

	  private:
		Vec3f _recursiveLighting( const Scene & p_scene,
								  const Ray &	p_ray,
								  HitRecord &	p_hitRecord,
								  const float	p_tMin,
								  const float	p_tMax,
								  int			bounce,
								  bool			isInMaterial ) const;
		Vec3f _getSkyColor( const Ray & p_ray ) const;
		Vec3f _toneMap(const Vec3f & c ) const;
		Vec3f sunDirectionFromAngles( float elevation, float azimuth )
		{
			elevation  = degToRad( elevation );
			azimuth	   = degToRad( azimuth );
			float cosE = cos( elevation );

			return normalize( Vec3f( -cosE * cos( azimuth ), sin( elevation ), -cosE * sin( azimuth ) ) );
		}
		// Number of bounces for the recursive ray tracing
		int _nbBounces = 5;
		Vec3f		_sunDirection				  = sunDirectionFromAngles(70.f, 20.f);
		int			_skyColorSamples			  = 32;
        const float _hr = 8.0f;
		const float _hm			 = 1.2f;
		const Vec3f _betaR		 = Vec3f( 3.8e-6f, 13.5e-6f, 33.1e-6f ) * 1000.f;
		const Vec3f _betaM		 = Vec3f( 21e-6f ) * 1000.f;
		const float	_exposure					  = 200.f;

	};
} // namespace RT
#endif // __RT_ISICG_WHITTED_INTEGRATOR__