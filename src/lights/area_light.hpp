#ifndef __RT_ISICG_AREA_LIGHT__
#define __RT_ISICG_AREA_LIGHT__

#include "base_light.hpp"
#include "materials/emissive_material.hpp"
#include "objects/triangle_mesh.hpp"
#include "ray.hpp"
#include "scene.hpp"
#include <random>

namespace RT
{
	inline TriangleMeshGeometry getRandomTriangle( const std::vector<TriangleMeshGeometry> & meshGeometry )
	{
		static std::mt19937					  gen( std::random_device {}() );
		std::uniform_int_distribution<size_t> dist( 0, meshGeometry.size() - 1 );
		return meshGeometry[ dist( gen ) ];
	}
	// This class allows an object to be used as a light source
	class AreaLight : public BaseLight
	{
	  public:
		AreaLight( const std::string & p_name, const MeshTriangle * p_mesh, const EmissiveMaterial * p_material )
			: BaseLight( p_name, p_material->getFlatColor(), p_material->getIntensity() ), _mesh( p_mesh ),
			  material( p_material )
		{
			_isSurface = true;
		}

		~AreaLight() = default;

		LightSample sample( const Vec3f & p_point ) const;

	  private:
		const MeshTriangle *	 _mesh;	   // object that we want to be considered as a light
		const EmissiveMaterial * material; // emissive material
	};
} // namespace RT

#endif // __RT_ISICG_AREA_LIGHT__