#ifndef __RT_ISICG_BASE_LIGHT_
#define __RT_ISICG_BASE_LIGHT_

#include "defines.hpp"
#include "light_sample.hpp"

namespace RT
{
	enum LightType{
		POINT,
		CYLINDER,
		DIRECTIONNAL,
		QUAD
	};

	class BaseLight
	{
	  public:
		BaseLight( const std::string p_name, const Vec3f & p_color, const float p_power = 1.f )
			: _name( p_name ), _color( p_color ), _power( p_power )
		{
		}
		virtual ~BaseLight() = default;

		inline const Vec3f & getFlatColor() const { return _color; }
		inline float getPower() const {return _power;}
		virtual LightSample sample( const Vec3f & p_point ) const = 0;
		inline bool			isSurface() const { return _isSurface; }
		virtual LightType getType() const = 0;
		

	  protected:
		std::string _name;
		Vec3f		_color = WHITE;
		float		_power = 1.f;
		bool		_isSurface;
	};

} // namespace RT

#endif // __RT_ISICG_BASE_LIGHT__
