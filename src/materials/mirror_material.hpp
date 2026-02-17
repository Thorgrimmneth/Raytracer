#ifndef __RT_ISICG_MIRROR_MATERIAL__
#define __RT_ISICG_MIRROR_MATERIAL__

#include "base_material.hpp"
#include "brdfs/lambert_brdf.hpp"

namespace RT
{
	class MirrorMaterial : public BaseMaterial
	{
	  public:
		MirrorMaterial( const std::string & p_name ) : BaseMaterial( p_name ), _brdf( BLACK ) {}
		MirrorMaterial( const std::string & p_name, const Vec3f & p_diffuse )
			: BaseMaterial( p_name ), _brdf( p_diffuse )
		{
		}

		virtual ~MirrorMaterial() = default;

		Vec3f shade( const Ray &		 p_ray,
					 const HitRecord &	 p_hitRecord,
					 const LightSample & p_lightSample ) const override
		{
			return BLACK;
		}

		inline const Vec3f & getFlatColor() const override { return BLACK; }

		const bool isMirror() const override { return true; }

		MaterialType getType() const override { return MaterialType::MIRROR;}

	  protected:
		LambertBRDF _brdf;
	};

} // namespace RT

#endif // __RT_ISICG_MIRROR_MATERIAL__
