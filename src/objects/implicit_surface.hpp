#ifndef __RT_ISICG_IMPLICIT_SURFACE__
#define __RT_ISICG_IMPLICIT_SURFACE__

#include "base_object.hpp"

namespace RT
{

	class ImplicitSurface : public BaseObject
	{
	  public:
		ImplicitSurface()		   = delete;
		virtual ~ImplicitSurface() = default;

		ImplicitSurface( const std::string & p_name ) : BaseObject( p_name ) {}

		// Check for nearest intersection between p_tMin and p_tMax : if found fill p_hitRecord.
		virtual bool intersect( const Ray & p_ray,
								const float p_tMin,
								const float p_tMax,
								HitRecord & p_hitRecord ) const override;

		// Check for any intersection between p_tMin and p_tMax.
		virtual bool intersectAny( const Ray & p_ray, const float p_tMin, const float p_tMax ) const override;

		virtual Vec3f evaluateNormal( const Vec3f & p_point ) const
		{
			float e = 1.e-6f;
			return normalize( Vec3f( 1.f, -1.f, -1.f ) * _sdf( p_point + Vec3f( e, -e, -e ) )
							  + Vec3f( -1.f, -1.f, 1.f ) * _sdf( p_point + Vec3f( -e, -e, e ) )
							  + Vec3f( -1.f, 1.f, -1.f ) * _sdf( p_point + Vec3f( -e, e, -e ) )
							  + Vec3f( 1.f, 1.f, 1.f ) * _sdf( p_point + Vec3f( e, e, e ) ) );
		}

		virtual void lookAt( const Vec3f & target, const Vec3f & _position );
		virtual void rotateFromEuler( const Vec3f & angles );

		virtual void lookAtThenEuler( const Vec3f & target, const Vec3f & _position, const Vec3f & angles );
		Mat3f		 getRotationMatrix() const { return _rotationMatrix; }
		ObjectType getType() const override { return ObjectType::Implicit;}
	  private:
		virtual float _sdf( const Vec3f & p_point ) const = 0;

		const float _minDistance	= 1e-3f;
		Mat3f		_rotationMatrix = MAT3F_ID;
	};

} // namespace RT

#endif // __RT_ISICG_IMPLICIT_SURFACE__
