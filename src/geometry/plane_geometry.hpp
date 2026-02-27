#ifndef __RT_ISICG_PLANE_GEOMETRY__
#define __RT_ISICG_PLANE_GEOMETRY__

#include "base_geometry.hpp"

namespace RT
{

	class PlaneGeometry : public BaseGeometry
	{
	  public:
		PlaneGeometry( const Vec3f & p_position, const Vec3f & p_normal ) : _position( p_position )
		{
			_normal = normalize(p_normal);
			_delta = glm::dot( -_normal, _position );
		}

		~PlaneGeometry() = default;

		inline const Vec3f getNormal() const { return normalize( _normal ); }
		inline const Vec3f & getPosition() const { return _position; }
		inline float getDelta() const {return _delta;}

		bool intersect( const Ray & p_ray, float & p_t1 ) const;

	  private:
		float _delta;
		Vec3f _normal;
		Vec3f _position;
	};
} // namespace RT

#endif // __RT_ISICG_PLANE_GEOMETRY__