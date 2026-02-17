#include "point_light.hpp"

namespace RT
{
	LightSample PointLight::sample( const Vec3f & p_point ) const
	{
		float distance = glm::distance( p_point, _position );
		Vec3f radiance = _color * _power / ( distance * distance );

		LightSample rep;
		rep._radiance  = radiance;
		rep._pdf	   = 1.f;
		rep._power	   = _power;
		rep._distance  = distance;
		rep._direction = glm::normalize( _position - p_point );

		return rep;
	}

} // namespace RT