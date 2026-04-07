#ifndef __RT_ISICG_COLOR_MATERIAL__
#define __RT_ISICG_COLOR_MATERIAL__

#include "base_material.hpp"
#include "defines.hpp"

namespace RT
{
	class ColorMaterial : public BaseMaterial
	{
	  public:
		ColorMaterial() : BaseMaterial( "default" ) {}
		ColorMaterial( const std::string & p_name, const Vec3f & p_color ) : BaseMaterial( p_name ), _color( p_color )
		{
		}
		ColorMaterial( const std::string & p_name ) : BaseMaterial( p_name ) 
		{
			_color = Vec3f( randomFloat(), randomFloat(), randomFloat() );
		}

		virtual ~ColorMaterial() = default;

		Vec3f shade( const Ray & p_ray, const HitRecord & p_hit, const LightSample & p_lightSample ) const override
		{
			return _color;
		}

		inline const Vec3f & getFlatColor() const override { return _color; }

		MaterialType getType() const override { return MaterialType::COLOR;}

	  protected:
		Vec3f _color = Vec3f( 1.f, 1.f, 1.f );
	};

} // namespace RT

#endif // __RT_ISICG_COLOR_MATERIAL__
