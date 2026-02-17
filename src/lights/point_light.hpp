#ifndef __RT_ISICG_POINT_LIGHT__
#define __RT_ISICG_POINT_LIGHT__

#include "base_light.hpp"

namespace RT
{
	class PointLight : public BaseLight
	{
	  public:
		PointLight( const std::string p_name, const Vec3f & p_pos, const Vec3f & p_color, const float p_power = 1.f )
			: BaseLight( p_name, p_color, p_power ), _position( p_pos )
		{
		}
		virtual ~PointLight() = default;

		LightSample sample( const Vec3f & p_point ) const override;
		LightType getType() const override { return LightType::POINT;}
		Vec3f getPosition() const { return _position;}
	  private:
		Vec3f _position;
	};
} // namespace RT
#endif // __RT_ISICG_POINT_LIGHT__