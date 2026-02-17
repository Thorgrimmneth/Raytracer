#include "implicit_surface.hpp"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

namespace RT
{
	// Sphere-tracing
	bool ImplicitSurface::intersect( const Ray & p_ray,
									 const float p_tMin,
									 const float p_tMax,
									 HitRecord & p_hitRecord ) const
	{
		float tMin = p_tMin;
		int	  cpt  = 0;
		while ( tMin < p_tMax && cpt < 1000 )
		{
			const Vec3f point = p_ray.pointAtT( tMin );
			const float dist  = _sdf( point );

			if ( dist < _minDistance )
			{
				p_hitRecord._distance = tMin;
				p_hitRecord._point	  = point;
				p_hitRecord._normal	  = evaluateNormal( point );
				p_hitRecord._object	  = this;

				return true;
			}

			tMin += dist;
			cpt += 1;
		}
		return false;
	}

	bool ImplicitSurface::intersectAny( const Ray & p_ray, const float p_tMin, const float p_tMax ) const
	{
		HitRecord hitRecord;
		if ( intersect( p_ray, p_tMin, p_tMax, hitRecord ) )
		{
			if ( hitRecord._distance < p_tMax && hitRecord._distance > p_tMin ) { return true; }
			return false;
		}
		return false;
	}

	// Modifies the _rotationMatrix to look at the target point from the position point (center of the object)
	void ImplicitSurface::lookAt( const Vec3f & target, const Vec3f & _position )
	{
		Vec3f forward = normalize( target - _position );
		Vec3f up( 0.f, 1.f, 0.f );

		Vec3f right	 = glm::normalize( glm::cross( up, forward ) );
		Vec3f realUp = glm::cross( forward, right );

		_rotationMatrix = glm::mat3( right, realUp, forward );
	}

	// Modifies the _rotationMatrix according to the euler angles (pitch, yaw, roll)
	void ImplicitSurface::rotateFromEuler( const Vec3f & angles )
	{
		// angles : pitch (X), yaw (Y), roll (Z)
		glm::quat rotationQuat = glm::quat( angles );
		_rotationMatrix		   = glm::toMat3( rotationQuat );
	}

	// Modifies the _rotationMatrix with the 2 techniques used above
	void ImplicitSurface::lookAtThenEuler( const Vec3f & target, const Vec3f & position, const Vec3f & eulerAngles )
	{
		Vec3f	  forward  = glm::normalize( target - position );
		glm::quat lookQuat = glm::rotation( Vec3f( 0.f, 0.f, 1.f ), forward );

		glm::quat eulerQuat = glm::quat( eulerAngles );

		glm::quat final = lookQuat * eulerQuat;
		_rotationMatrix = glm::toMat3( final );
	}
} // namespace RT
