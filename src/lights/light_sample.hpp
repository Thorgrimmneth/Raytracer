#ifndef __RT_ISICG_LIGHT_SAMPLE__
#define __RT_ISICG_LIGHT_SAMPLE__

#include "defines.hpp"

namespace RT
{
	struct LightSample
	{
	  public:
		LightSample() = default;
		LightSample( const Vec3f & p_direction,
					 const float   p_distance,
					 const Vec3f & p_radiance,
					 const float   p_pdf,
					 const float   p_power )
			: _direction( p_direction ), _distance( p_distance ), _radiance( p_radiance ), _pdf( p_pdf ),
			  _power( p_power )
		{
		}

		Vec3f _direction = VEC3F_ZERO;
		float _distance	 = 0.f;
		Vec3f _radiance	 = BLACK;
		float _pdf		 = 1.f;
		float _power	 = 1.f;
	};

} // namespace RT

#endif // __RT_ISICG_LIGHT_SAMPLE__
