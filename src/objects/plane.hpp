#ifndef __RT_ISICG_PLANE__
#define __RT_ISICG_PLANE__

#include "base_object.hpp"
#include "geometry/plane_geometry.hpp"

namespace RT
{

	class Plane : public BaseObject
	{
	  public:
		Plane() = default;
		Plane( const std::string & p_name, const Vec3f & p_position, const Vec3f & p_normal )
			: BaseObject( p_name ), _plane_geometry( p_position, p_normal )
		{
		}

		~Plane() = default;

		virtual bool intersect( const Ray & p_ray,
								const float p_tMin,
								const float p_tMax,
								HitRecord & p_hitRecord ) const override;

		virtual bool intersectAny( const Ray & p_ray, const float p_tMin, const float p_tMax ) const override;
		ObjectType getType() const override { return ObjectType::Plane;}
		const PlaneGeometry& getGeometry() const {return _plane_geometry;}
	  private:
		std::string	  _name;
		PlaneGeometry _plane_geometry;
	};
} // namespace RT

#endif // __RT_ISICG_PLANE_GEOMETRY__