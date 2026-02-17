#ifndef __RT_ISICG_IMPLICIT_TORUS__
#define __RT_ISICG_IMPLICIT_TORUS__

#include "implicit_surface.hpp"

namespace RT
{
	class ImplicitTorus : public ImplicitSurface
	{
	  public:
		ImplicitTorus( const std::string & p_name, const Vec3f & p_center, const Vec2f & p_radius )
			: ImplicitSurface( p_name ), _center( p_center ), _radius( p_radius )
		{
			if ( _radius.x < _radius.y ) std::swap( _radius.x, _radius.y );
		}

		ImplicitTorus( const std::string &	   p_name,
					   const Vec3f &		   p_center,
					   const Vec2f &		   p_radius,
					   const Vec3f &		   p_target,
					   const OrientationMode & p_mode,
					   const Vec3f &		   p_angles = Vec3f( 0.f ) )
			: ImplicitSurface( p_name ), _center( p_center ), _radius( p_radius )
		{
			_rotated = true;
			if ( p_mode == OrientationMode::LookAt )
				lookAt( p_target, p_center );
			else if ( p_mode == OrientationMode::Euler )
				rotateFromEuler( p_target );
			else if ( p_mode == OrientationMode::LookAtThenEuler )
				lookAtThenEuler( p_target, p_center, p_angles );
		};

		~ImplicitTorus() = default;
		ObjectType getType() const override { return ObjectType::Implicit;}
	  private:
		virtual float _sdf( const Vec3f & p_point ) const override
		{
			Vec3f pLocal = p_point - _center;
			if ( _rotated == true )
			{
				Mat3f invRot = glm::transpose( getRotationMatrix() );
				pLocal		 = invRot * pLocal;
			}

			float distTemp = glm::length( Vec2f( pLocal.x, pLocal.y ) ) - _radius.x;
			return glm::length( Vec2f( distTemp, pLocal.z ) ) - _radius.y;
		}

		Vec3f _center;
		Vec2f _radius;
		bool  _rotated = false;
	};

} // namespace RT

#endif // __RT_ISICG_IMPLICIT_TORUS__
