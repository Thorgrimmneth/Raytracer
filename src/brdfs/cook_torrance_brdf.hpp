#ifndef __RT_ISICG_BRDF_COOK_TORRANCE__
#define __RT_ISICG_BRDF_COOK_TORRANCE__

#include "defines.hpp"

namespace RT
{
	class CookTorranceBRDF
	{
	  public:
		CookTorranceBRDF() : _ks( WHITE ), _ruggedness( 0.5f ) { _alpha = _ruggedness * _ruggedness; };
		CookTorranceBRDF( const Vec3f & p_ks, const float & p_ruggedness ) : _ks( p_ks ), _ruggedness( p_ruggedness )
		{
			_alpha = _ruggedness * _ruggedness;
		};

		float computeD( const Vec3f & p_normal, const Vec3f & h ) const
		{
			float alphaSquared = _alpha * _alpha;
			float NdotH		   = glm::dot( p_normal, h );
			return alphaSquared / ( PIf * pow( ( NdotH * NdotH ) * ( alphaSquared - 1.f ) + 1.f, 2.f ) );
		}

		Vec3f computeF( const Vec3f & wo, const Vec3f & h ) const
		{
			float HdotV = glm::clamp( glm::dot( h, wo ), 0.0f, 1.0f );
			return _ks + ( Vec3f( 1.f ) - _ks ) * glm::pow( 1.f - HdotV, 5.f );
		}

		float computeG1( const float & x, const float & k ) const { return x / ( x * ( 1.f - k ) + k ); }
		float computeG( const Vec3f & wi, const Vec3f & wo, const Vec3f & p_normal ) const
		{
			float k = glm::pow( ( _ruggedness + 1.f ), 2.f ) / 8.f;
			return computeG1( glm::dot( p_normal, wo ), k ) * computeG1( glm::dot( p_normal, wi ), k );
		}

		inline Vec3f evaluate( const Vec3f & p_ray, const Vec3f & p_normal, const Vec3f & p_direction ) const
		{
			Vec3f wo = -p_ray;
			Vec3f wi = p_direction;

			float D = computeD( p_normal, glm::normalize( wi + wo ) );
			Vec3f F = computeF( wo, glm::normalize( wi + wo ) );
			float G = computeG( wi, wo, p_normal );

			float denominator = 4.f * glm::dot( wo, p_normal ) * glm::dot( wi, p_normal );
			return _ks * ( ( D * F * G ) / denominator );
		}

		inline const Vec3f & getKs() const { return _ks; }
		inline const float getR() const {return _ruggedness;}

	  private:
		Vec3f _ks		  = BLACK;
		float _ruggedness = 0.5f;
		float _alpha	  = 0.25f;
	};
} // namespace RT

#endif // __RT_ISICG_BRDF_COOK_TORRANCE__
