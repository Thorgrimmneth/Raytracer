#include "directionnal_light.hpp"

namespace RT
{
	LightSample DirectionnalLight::sample( const Vec3f & p_point ) const
	{

		LightSample rep;
		rep._radiance  = _color;
		rep._pdf	   = 1.f;
		rep._power	   = _power;
		rep._distance  = 3000.f;
		rep._direction = -_direction;

		return rep;
	}

} // namespace RT