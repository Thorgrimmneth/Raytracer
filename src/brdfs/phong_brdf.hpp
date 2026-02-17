#ifndef __RT_ISICG_BRDF_PHONG__
#define __RT_ISICG_BRDF_PHONG__

#include "defines.hpp"

namespace RT
{
	class PhongBRDF
	{
	  public:
		PhongBRDF( const Vec3f & p_ks, const float & p_s ) : _ks( p_ks ), _s( p_s ) {};

		inline Vec3f evaluate( const Vec3f & p_ray, const Vec3f & p_normal, const Vec3f & p_direction ) const
		{
			// formula from moteur3D course
			Vec3f wi		= p_direction;
			Vec3f wo		= -p_ray;
			Vec3f wr		= glm::reflect( -wi, p_normal );
			float cosThetaI = glm::max( glm::dot( p_normal, wi ), 0.001f ); // prevent division by 0
			float cosAlpha	= glm::max( glm::dot( glm::normalize( wr ), glm::normalize( wo ) ), 0.0f );
			float cosAlphaS = glm::pow( cosAlpha, _s );

			return _ks / cosThetaI * cosAlphaS;
		}

		inline const Vec3f & getKs() const { return _ks; }

	  private:
		// specular color
		Vec3f _ks = BLACK;
		// shininess
		float _s = 8.f;
	};
} // namespace RT

#endif // __RT_ISICG_BRDF_PHONG__
