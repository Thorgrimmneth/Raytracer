#ifndef __RT_ISICG_BASE_MATERIAL__
#define __RT_ISICG_BASE_MATERIAL__

#include "hit_record.hpp"
#include "lights/light_sample.hpp"
#include "ray.hpp"

namespace RT
{
	enum class MaterialType
	{
		COLOR,
		EMISSIVE,
		LAMBERT,
		METAL,
		MIRROR,
		PLASTIC,
		TRANSPARENT
	};

	class BaseMaterial
	{
	  public:
		BaseMaterial() = delete;
		BaseMaterial( const std::string & p_name ) : _name( p_name ) {}
		virtual ~BaseMaterial() = default;

		virtual Vec3f shade( const Ray & p_ray, const HitRecord & p_hitRecord, const LightSample & p_lightSample ) const
			= 0;

		virtual inline const Vec3f & getFlatColor() const = 0;

		virtual const std::string & getName() const final { return _name; }

		virtual const bool isMirror() const { return false; }

		virtual const bool isTransparent() const { return false; }

		virtual const float getIOR() const { return 1.f; }

		virtual const float getOpacity() const { return 1.f; }

		virtual MaterialType getType() const = 0;

	  protected:
		std::string _name;
	};

} // namespace RT

#endif // __RT_ISICG_BASE_MATERIAL__
