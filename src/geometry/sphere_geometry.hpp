#ifndef __RT_ISICG_SPHERE_GEOMETRY__
#define __RT_ISICG_SPHERE_GEOMETRY__

#include "base_geometry.hpp"

namespace RT
{
	class SphereGeometry : public BaseGeometry
	{
	  public:
		SphereGeometry()		  = delete;
		virtual ~SphereGeometry() = default;

		SphereGeometry( const Vec3f & p_center1, const float p_radius )
			: _center( p_center1, VEC3F_ZERO ), _radius( p_radius ), _centerTemp( p_center1 )
		{
		}

		SphereGeometry( const Vec3f & p_center1, const Vec3f & p_center2, const float p_radius )
			: _center( p_center1, p_center2 - p_center1 ), _radius( p_radius ), _centerTemp( p_center1 )
		{
		}

		inline const Vec3f getCenter(const double time = 0) const { return _center.pointAtT(time); }
		inline float	 getRadius() const { return _radius; }
		inline const Ray getCenters() const {
			return _center;
		}
		bool intersect( const Ray & p_ray, float & p_t1, float & p_t2 ) const;

		inline Vec3f computeNormal( const Vec3f & p_point, const double time = 0 ) const
		{
			return normalize( _center.pointAtT( time ) - p_point );
		}

	  private:
		Ray _center;
		Vec3f _centerTemp;
		float _radius = 1.f;
	};

} // namespace RT

#endif // __RT_ISICG_SPHERE_GEOMETRY__
