#ifndef __RT_ISICG_DIRECTIONNAL_LIGHT__
#define __RT_ISICG_DIRECTIONNAL_LIGHT__

#include "base_light.hpp"

namespace RT
{
	class DirectionnalLight : public BaseLight
	{
	  public:
		DirectionnalLight( const std::string p_name,
						   const Vec3f &	 p_direction,
						   const Vec3f &	 p_color,
						   const float		 p_power = 1.f )
			: BaseLight( p_name, p_color, p_power ), _direction( p_direction )
		{
		}
		virtual ~DirectionnalLight() = default;

		LightSample sample( const Vec3f & p_point ) const override;
		LightType getType() const override { return LightType::DIRECTIONNAL;}
		Vec3f getDirection() const { return _direction;}
	  private:
		Vec3f _direction;
	};
} // namespace RT
#endif // __RT_ISICG_DIRECTIONNAL_LIGHT__