#ifndef __RT_ISICG_EMISSIVE_MATERIAL__
#define __RT_ISICG_EMISSIVE_MATERIAL__

#include "base_material.hpp"

namespace RT
{
	class EmissiveMaterial : public BaseMaterial
	{
	  public:
		EmissiveMaterial( const std::string & p_name,
						  const Vec3f &		  p_color	  = Vec3f( 1.f, 1.f, 1.f ),
						  const float		  p_intensity = 1.f )
			: BaseMaterial( p_name ), _color( p_color ), _intensity( p_intensity )
		{
		}

		~EmissiveMaterial() = default;

		Vec3f shade( const Ray &		 p_ray,
					 const HitRecord &	 p_hitRecord,
					 const LightSample & p_lightSample ) const override
		{
			return _color
				   * _intensity; // Emissive materials create light, so we return the color multiplied by intensity (can be more than 1)
		}

		inline const Vec3f & getFlatColor() const override { return _color; }

		inline const Vec3f getEmissionColor() const { return _color * _intensity; }

		inline const bool isEmissive() const { return true; }

		inline const float & getIntensity() const { return _intensity; }

		MaterialType getType() const override { return MaterialType::EMISSIVE;}

	  private:
		Vec3f _color;
		float _intensity;
	};
} // namespace RT

#endif // __RT_ISICG_EMISSIVE_MATERIAL__