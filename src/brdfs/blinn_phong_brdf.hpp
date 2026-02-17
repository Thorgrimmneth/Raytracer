#ifndef __RT_ISICG_BRDF_BLINN_PHONG__
#define __RT_ISICG_BRDF_BLINN_PHONG__

#include "defines.hpp"

namespace RT
{
	class BlinnPhongBRDF
	{
	  public:
		BlinnPhongBRDF( const Vec3f & p_ks, const float & p_s ) : _ks( p_ks ), _s( p_s ) {};

		inline Vec3f evaluate( const Vec3f & p_ray, const Vec3f & p_normal, const Vec3f & p_direction ) const
		{
			// formula from moteur3D course
			Vec3f wo = -p_ray;
			Vec3f wi = p_direction;
			// half vector
			Vec3f wh = glm::normalize( wi + wo );

			float cosAlpha = glm::max( 0.f, glm::dot( p_normal, wh ) );
			return _ks * glm::pow( cosAlpha, _s );
		}

		inline const Vec3f & getKs() const { return _ks; }

	  private:
		// specular color
		Vec3f _ks = BLACK;
		// shininess
		float _s = 8.f;
	};
} // namespace RT

#endif // __RT_ISICG_BRDF_BLINN_PHONG__
