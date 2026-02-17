#ifndef __RT_ISICG_IMPLICIT_SPHERE__
#define __RT_ISICG_IMPLICIT_SPHERE__

#include "implicit_surface.hpp"

namespace RT
{

	class ImplicitSphere : public ImplicitSurface
	{
	  public:
		ImplicitSphere( const std::string & p_name, const Vec3f & p_center, const float & p_radius )
			: ImplicitSurface( p_name ), _center( p_center ), _radius( p_radius )
		{
		}
		ObjectType getType() const override { return ObjectType::Implicit;}
		~ImplicitSphere() = default;

	  private:
		virtual float _sdf( const Vec3f & p_point ) const
		{
			float dist = glm::length( p_point - _center ) - ( _radius );
			return dist;
		};

		Vec3f _center;
		float _radius;
	};
} // namespace RT

#endif // __RT_ISICG_IMPLICIT_SPHERE__