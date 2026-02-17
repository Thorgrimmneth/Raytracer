#ifndef __RT_ISICG_LAMBERT_MATERIAL__
#define __RT_ISICG_LAMBERT_MATERIAL__

#include "base_material.hpp"
#include "brdfs/lambert_brdf.hpp"

namespace RT
{
	class LambertMaterial : public BaseMaterial
	{
	  public:
		LambertMaterial( const std::string & p_name, const Vec3f & p_diffuse )
			: BaseMaterial( p_name ), _brdf( p_diffuse )
		{
		}

		virtual ~LambertMaterial() = default;

		Vec3f shade( const Ray &		 p_ray,
					 const HitRecord &	 p_hitRecord,
					 const LightSample & p_lightSample ) const override
		{
			return _brdf.evaluate();
		}

		inline const Vec3f & getFlatColor() const override { return _brdf.getKd(); }

		MaterialType getType() const override { return MaterialType::LAMBERT;}

	  protected:
		LambertBRDF _brdf;
	};

} // namespace RT

#endif // __RT_ISICG_LAMBERT_MATERIAL__
