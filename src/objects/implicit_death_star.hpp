#ifndef __RT_ISICG_IMPLICIT_DEATH_STAR__
#define __RT_ISICG_IMPLICIT_DEATH_STAR__

#include "implicit_surface.hpp"

namespace RT
{
	class ImplicitDeathStar : public ImplicitSurface
	{
	  public:
		ImplicitDeathStar( const std::string & p_name,
						   const Vec3f &	   p_center,
						   const float &	   p_ra,
						   const float &	   p_rb,
						   const float &	   p_d )
			: ImplicitSurface( p_name ), _center( p_center ), _r1( p_ra ), _r2( p_rb ), _d( p_d ) {};

		// Constructor used to create a Death Star with a specific orientation
		ImplicitDeathStar( const std::string &	   p_name,
						   const Vec3f &		   p_center,
						   const float &		   p_ra,
						   const float &		   p_rb,
						   const float &		   p_d,
						   const Vec3f &		   p_target,
						   const OrientationMode & p_mode,
						   const Vec3f &		   p_angles = Vec3f( 0.f ) )
			: ImplicitSurface( p_name ), _center( p_center ), _r1( p_ra ), _r2( p_rb ), _d( p_d )
		{
			if ( p_mode == OrientationMode::LookAt )
				lookAt( p_target, p_center );
			else if ( p_mode == OrientationMode::Euler )
				rotateFromEuler( p_target );
			else if ( p_mode == OrientationMode::LookAtThenEuler )
				lookAtThenEuler( p_target, p_center, p_angles );
		};
		ObjectType getType() const override { return ObjectType::Implicit;}
		~ImplicitDeathStar() = default;

	  private:
		virtual float _sdf( const Vec3f & p_point ) const override
		{
			Mat3f invRot = glm::transpose( getRotationMatrix() );
			Vec3f pLocal = invRot * ( p_point - _center );

			Vec2f p = Vec2f( pLocal.x, glm::length( Vec2f( pLocal.y, pLocal.z ) ) );

			float a = ( _r1 * _r1 - _r2 * _r2 + _d * _d ) / ( 2.f * _d );
			float b = glm::sqrt( glm::max( _r1 * _r1 - a * a, 0.f ) );
			if ( p.x * b - p.y * a > _d * glm::max( b - p.y, 0.f ) ) { return glm::length( p - Vec2f( a, b ) ); }
			else { return glm::max( ( length( p ) - _r1 ), -( length( p - Vec2f( _d, 0.f ) ) - _r2 ) ); }
		}

		Vec3f _center;
		float _r1;
		float _r2;
		float _d;
	};

} // namespace RT

#endif // __RT_ISICG_IMPLICIT_DEATH_STAR__
