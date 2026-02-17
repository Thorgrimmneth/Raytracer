#include "plane_geometry.hpp"

namespace RT
{

	bool PlaneGeometry::intersect( const Ray & p_ray, float & p_t1 ) const
	{
		float ND = glm::dot( _normal, p_ray.getDirection() );

		// If the ray is parallel to the plane, there is no intersection
		if ( glm::abs( ND ) <= 1.0e-6f ) { return false; }

		float numerator = -( _delta + glm::dot( _normal, p_ray.getOrigin() ) );

		p_t1 = numerator / ND;

		return true;
	}
} // namespace RT