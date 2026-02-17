#ifndef __RT_ISICG_SCENE__
#define __RT_ISICG_SCENE__

#include "defines.hpp"
#include "lights/base_light.hpp"
#include "objects/base_object.hpp"
#include <exception>
#include <map>
#include <vector>

namespace RT
{
	using ObjectMap		  = std::map<const std::string, BaseObject *>;
	using ObjectMapPair	  = ObjectMap::value_type;
	using MaterialMap	  = std::map<const std::string, BaseMaterial *>;
	using MaterialMapPair = MaterialMap::value_type;
	using LightList		  = std::vector<BaseLight *>;

	class Scene
	{
	  public:
		Scene();
		~Scene();

		// Hard coded initialization.
		void init();

		// Initialization from file.
		void init( const std::string & p_path ) { throw std::logic_error("Not implemented!"); }

		void loadFileTriangleMesh( const std::string & p_name, const std::string & p_path );

		const LightList & getLights() const { return _lightList; }

		// Check for nearest intersection between p_tMin and p_tMax : if found fill p_hitRecord.
		bool intersect( const Ray & p_ray, const float p_tMin, const float p_tMax, HitRecord & p_hitRecord ) const;
		bool intersectAny( const Ray & p_ray, const float p_tMin, const float p_tMax ) const;

		inline const ObjectMap& getObject() const{return _objectMap;}
		inline const MaterialMap& getMaterial() const{return _materialMap;}
		inline const LightList& getLight() const{return _lightList;}

	  private:
		void _addObject( BaseObject * p_object );
		void _addMaterial( BaseMaterial * p_material );
		void _addLight( BaseLight * p_light );

		void _attachMaterialToObject( const std::string & p_materialName, const std::string & p_objectName );
		void _tp3();
		void _tp4Bunny();
		void _tp4Conference();
		void _tp5();
		void _tp6();
		void _killTheBunny();
		void _DOF();
		void _sphere();
		void _spheres();

	  private:
		ObjectMap	_objectMap;
		MaterialMap _materialMap;
		LightList	_lightList;
	};
} // namespace RT

#endif // __RT_ISICG_SCENE__
