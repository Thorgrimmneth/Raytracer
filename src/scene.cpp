#include "scene.hpp"
#include "lights/area_light.hpp"
#include "lights/cylinder_light.hpp"
#include "lights/point_light.hpp"
#include "lights/quad_light.hpp"
#include "lights/directionnal_light.hpp"
#include "materials/color_material.hpp"
#include "materials/emissive_material.hpp"
#include "materials/lambert_material.hpp"
#include "materials/metal_material.hpp"
#include "materials/mirror_material.hpp"
#include "materials/plastic_material.hpp"
#include "materials/transparent_material.hpp"
#include "objects/implicit_death_star.hpp"
#include "objects/implicit_line.hpp"
#include "objects/implicit_sphere.hpp"
#include "objects/implicit_torus.hpp"
#include "objects/plane.hpp"
#include "objects/sphere.hpp"
#include "objects/triangle_mesh.hpp"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include "utils/random.hpp"


namespace RT
{
	Scene::Scene() { _addMaterial( new ColorMaterial( "default", WHITE ) ); }

	Scene::~Scene()
	{
		for ( const ObjectMapPair & object : _objectMap )
		{
			delete object.second;
		}
		for ( const MaterialMapPair & material : _materialMap )
		{
			delete material.second;
		}
		for ( const BaseLight * light : _lightList )
		{
			delete light;
		}
	}

	void Scene::init() { _spheres(); }

	void Scene::loadFileTriangleMesh( const std::string & p_name, const std::string & p_path )
	{
		std::cout << "Loading:" << p_path << std::endl;
		Assimp::Importer importer;

		// Read scene and triangulate meshes
		const aiScene * const scene
			= importer.ReadFile( p_path, aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_GenUVCoords );

		if ( scene == nullptr ) { throw std::runtime_error( "Fail to load file:" + p_path ); }

		unsigned int cptTriangles = 0;
		unsigned int cptVertices  = 0;

		for ( unsigned int m = 0; m < scene->mNumMeshes; ++m )
		{
			const aiMesh * const mesh = scene->mMeshes[ m ];
			if ( mesh == nullptr ) { throw std::runtime_error( "Fail to load file:" + p_path + ": mesh is null" ); }

			const std::string meshName = p_name + "_" + std::string( mesh->mName.C_Str() );
			std::cout << "-- Load mesh" << m + 1 << "/" << scene->mNumMeshes << ":" << meshName << std::endl;

			cptTriangles += mesh->mNumFaces;
			cptVertices += mesh->mNumVertices;

			const bool hasUV = mesh->HasTextureCoords( 0 );

			MeshTriangle * triMesh = new MeshTriangle( meshName );
			// Vertices before faces otherwise face normals cannot be computed.
			for ( unsigned int v = 0; v < mesh->mNumVertices; ++v )
			{
				triMesh->addVertex( mesh->mVertices[ v ].x, mesh->mVertices[ v ].y, mesh->mVertices[ v ].z );
				triMesh->addNormal( mesh->mNormals[ v ].x, mesh->mNormals[ v ].y, mesh->mNormals[ v ].z );
				if ( hasUV ) triMesh->addUV( mesh->mTextureCoords[ 0 ][ v ].x, mesh->mTextureCoords[ 0 ][ v ].y );
			}
			for ( unsigned int f = 0; f < mesh->mNumFaces; ++f )
			{
				const aiFace & face = mesh->mFaces[ f ];
				triMesh->addTriangle( face.mIndices[ 0 ], face.mIndices[ 1 ], face.mIndices[ 2 ] );
			}

			_addObject( triMesh );
			triMesh->buildBVH();
			const aiMaterial * const mtl = scene->mMaterials[ mesh->mMaterialIndex ];
			if ( mtl == nullptr )
			{
				std::cerr << "Material undefined," << meshName << "assigned to default material" << std::endl;
			}
			else
			{
				Vec3f kd = WHITE;
				Vec3f ks = BLACK;
				float s	 = 0.f;

				aiColor3D aiKd;
				if ( mtl->Get( AI_MATKEY_COLOR_DIFFUSE, aiKd ) == AI_SUCCESS ) kd = Vec3f( aiKd.r, aiKd.g, aiKd.b );
				aiColor3D aiKs;
				if ( mtl->Get( AI_MATKEY_COLOR_SPECULAR, aiKs ) == AI_SUCCESS ) ks = Vec3f( aiKs.r, aiKs.g, aiKs.b );
				float aiS = 0.f;
				if ( mtl->Get( AI_MATKEY_SHININESS, aiS ) == AI_SUCCESS ) s = aiS;
				aiString mtlName;
				mtl->Get( AI_MATKEY_NAME, mtlName );
				_addMaterial( new ColorMaterial( std::string( mtlName.C_Str() ), kd ) );
				_attachMaterialToObject( mtlName.C_Str(), meshName );
				//_addMaterial( new PlasticMaterial( std::string( mtlName.C_Str() ), kd, ks, s ) );
				//_attachMaterialToObject( mtlName.C_Str(), meshName );
			}

			std::cout << "-- [DONE]" << triMesh->getNbTriangles() << "triangles," << triMesh->getNbVertices()
					  << "vertices." << std::endl;
		}
		std::cout << "[DONE]" << scene->mNumMeshes << "meshes," << cptTriangles << "triangles," << cptVertices
				  << "vertices." << std::endl;
	}

	bool Scene::intersect( const Ray & p_ray, const float p_tMin, const float p_tMax, HitRecord & p_hitRecord ) const
	{
		float tMax = p_tMax;
		bool  hit  = false;
		for ( const ObjectMapPair & object : _objectMap )
		{
			if ( object.second->intersect( p_ray, p_tMin, tMax, p_hitRecord ) )
			{
				tMax = p_hitRecord._distance; // update tMax to conserve the nearest hit
				hit	 = true;
			}
		}
		return hit;
	}

	bool Scene::intersectAny( const Ray & p_ray, const float p_tMin, const float p_tMax ) const
	{
		for ( const ObjectMapPair & object : _objectMap )
		{
			if ( object.second->intersectAny( p_ray, p_tMin, p_tMax ) ) { return true; }
		}
		return false;
	}
	
	void Scene::_addObject( BaseObject * p_object )
	{
		const std::string & name = p_object->getName();
		if ( _objectMap.find( name ) != _objectMap.end() )
		{
			std::cout << "[Scene::addObject] Object \'" << name << "\' already exists" << std::endl;
			delete p_object;
		}
		else
		{
			_objectMap[ name ] = p_object;
			_objectMap[ name ]->setMaterial( _materialMap[ "default" ] );
		}
	}

	void Scene::_addMaterial( BaseMaterial * p_material )
	{
		const std::string & name = p_material->getName();
		if ( _materialMap.find( name ) != _materialMap.end() )
		{
			std::cout << "[Scene::_addMaterial] Material \'" << name << "\' already exists" << std::endl;
			delete p_material;
		}
		else
		{
			std::cout << "Material \'" << name << "\' added." << std::endl;
			_materialMap[ name ] = p_material;
		}
	}

	void Scene::_addLight( BaseLight * p_light ) { _lightList.emplace_back( p_light ); }

	void Scene::_attachMaterialToObject( const std::string & p_materialName, const std::string & p_objectName )
	{
		if ( _objectMap.find( p_objectName ) == _objectMap.end() )
		{
			std::cout << "[Scene::attachMaterialToObject] Object \'" << p_objectName << "\' does not exist"
					  << std::endl;
		}
		else if ( _materialMap.find( p_materialName ) == _materialMap.end() )
		{
			std::cout << "[Scene::attachMaterialToObject] Material \'" << p_materialName << "\' does not exist,"
					  << "object \'" << p_objectName << "\' keeps its material \'"
					  << _objectMap[ p_objectName ]->getMaterial()->getName() << "\'" << std::endl;
		}
		else { _objectMap[ p_objectName ]->setMaterial( _materialMap[ p_materialName ] ); }
	}

	void Scene::_tp3()
	{
		// Add objects.
		_addObject( new Sphere( "Sphere1", Vec3f( 0.f, 0.f, 3.f ), 1.f ) );

		_addObject( new Plane( "Plan1", Vec3f( 0.f, -2.f, 0.f ), Vec3f( 0.f, 1.f, 0.f ) ) );

		// Add materials.
		_addMaterial( new ColorMaterial( "Blue", BLUE ) );

		_addMaterial( new ColorMaterial( "Red", RED ) );

		// Add lights.
		_addLight( new QuadLight(
			"light_1", Vec3f( 1.f, 10.f, 1.f ), Vec3f( -2.f, 0.f, 0.f ), Vec3f( 0.f, 0.f, 2.f ), WHITE, 40.f ) );

		// Link objects and materials.
		_attachMaterialToObject( "Blue", "Sphere1" );
		_attachMaterialToObject( "Red", "Plan1" );
	}

	void Scene::_tp4Bunny()
	{
		// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		// = = = = = = = = =
		// Add materials .
		// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		// = = = = = = = = =
		_addMaterial( new ColorMaterial( "RedColor", RED ) );
		_addMaterial( new ColorMaterial( "GreenColor", GREEN ) );
		_addMaterial( new ColorMaterial( "BlueColor", BLUE ) );
		_addMaterial( new ColorMaterial( "GreyColor", GREY ) );
		_addMaterial( new ColorMaterial( "MagentaColor", MAGENTA ) );
		_addMaterial( new ColorMaterial( "YellowColor", YELLOW ) );
		_addMaterial( new ColorMaterial( "CyanColor", CYAN ) );
		// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		// = = = = = = = = = Add objects . = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		// = = = = = = = = = = = = = = = = = = = = = = = = =
		// OBJ.
		loadFileTriangleMesh( "bunny", "data/bunny/Bunny.obj" ); // bunny
		_attachMaterialToObject( "CyanColor", "bunny_defaultobject" );
		// Pseudo Cornell box made with infinite planes .
		_addObject( new Plane( "PlaneGround", Vec3f( 0.f, -3.f, 0.f ), Vec3f( 0.f, 1.f, 0.f ) ) );
		_attachMaterialToObject( "GreyColor", "PlaneGround" );
		_addObject( new Plane( "PlaneLeft", Vec3f( 5.f, 0.f, 0.f ), Vec3f( -1.f, 0.f, 0.f ) ) );
		_attachMaterialToObject( "RedColor", "PlaneLeft" );
		_addObject( new Plane( "PlaneCeiling", Vec3f( 0.f, 7.f, 0.f ), Vec3f( 0.f, -1.f, 0.f ) ) );
		_attachMaterialToObject( "GreenColor", "PlaneCeiling" );
		_addObject( new Plane( "PlaneRight", Vec3f( -5.f, 0.f, 0.f ), Vec3f( 1.f, 0.f, 0.f ) ) );
		_attachMaterialToObject( "BlueColor", "PlaneRight" );
		_addObject( new Plane( "PlaneFront", Vec3f( 0.f, 0.f, 10.f ), Vec3f( 0.f, 0.f, -1.f ) ) );
		_attachMaterialToObject( "MagentaColor", "PlaneFront" );
		_addObject( new Plane( "PlaneRear", Vec3f( 0.f, 0.f, -10.f ), Vec3f( 0.f, 0.f, 1.f ) ) );
		_attachMaterialToObject( "YellowColor", "PlaneRear" );

		// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		// = = = = = = = = = Add lights . = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		// = = = = = = = = = = = = = = = = = = = = = = = = =
		_addLight( new PointLight( "light1", Vec3f( 0.f, 3.f, -5.f ), WHITE, 100.f ) );
	}

	void Scene::_tp4Conference()
	{
		loadFileTriangleMesh( "conference", "data/conference/conference.obj" ); // conference
		_addLight( new QuadLight( "Light1",
								  Vec3f( 900.0f, 600.0f, -300.0f ),
								  Vec3f( -800.0f, 0.0f, 0.0f ),
								  Vec3f( 0.0f, 0.0f, 300.0f ),
								  WHITE,
								  20.0f ) );
		//_addLight( new PointLight("light1", Vec3f( 900.0f, 600.0f, -300.0f ), WHITE, 10000.f ) );
	}

	void Scene::_tp5()
	{
		// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		// = = = = = = = = =
		// Add materials .
		// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		// = = = = = = = = =
		_addMaterial( new ColorMaterial( "WhiteColor", WHITE ) );
		_addMaterial( new ColorMaterial( "RedColor", RED ) );
		_addMaterial( new ColorMaterial( "GreenColor", GREEN ) );
		_addMaterial( new ColorMaterial( "BlueColor", BLUE ) );
		_addMaterial( new ColorMaterial( "GreyColor", GREY ) );
		_addMaterial( new ColorMaterial( "MagentaColor", MAGENTA ) );
		_addMaterial( new ColorMaterial( "CyanColor", CYAN ) );
		_addMaterial( new MirrorMaterial( "Mirror" ) );
		_addMaterial( new TransparentMaterial( "Transparent" ) );
		// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		// = = = = = = = = = Add objects . = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		// = = = = = = = = = = = = = = = = = = = = = = = = = Spheres .
		_addObject( new Sphere( "Sphere1", Vec3f( -2.f, 0.f, 3.f ), 1.5f ) );
		_attachMaterialToObject( "Mirror", "Sphere1" );
		_addObject( new Sphere( "Sphere2", Vec3f( 2.f, 0.f, 3.f ), 1.5f ) );
		_attachMaterialToObject( "Transparent", "Sphere2" );
		// Pseudo Cornell box made with infinite planes .
		_addObject( new Plane( "PlaneGround", Vec3f( 0.f, -3.f, 0.f ), Vec3f( 0.f, 1.f, 0.f ) ) );
		_attachMaterialToObject( "GreyColor", "PlaneGround" );
		_addObject( new Plane( "PlaneLeft", Vec3f( 5.f, 0.f, 0.f ), Vec3f( -1.f, 0.f, 0.f ) ) );
		_attachMaterialToObject( "RedColor", "PlaneLeft" );
		_addObject( new Plane( "PlaneCeiling", Vec3f( 0.f, 7.f, 0.f ), Vec3f( 0.f, -1.f, 0.f ) ) );
		_attachMaterialToObject( "GreenColor", "PlaneCeiling" );
		_addObject( new Plane( "PlaneRight", Vec3f( -5.f, 0.f, 0.f ), Vec3f( 1.f, 0.f, 0.f ) ) );
		_attachMaterialToObject( "BlueColor", "PlaneRight" );
		_addObject( new Plane( "PlaneFront", Vec3f( 0.f, 0.f, 10.f ), Vec3f( 0.f, 0.f, -1.f ) ) );
		_attachMaterialToObject( "MagentaColor", "PlaneFront" );
		// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		// = = = = = = = = = Add lights . = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		// = = = = = = = = = = = = = = = = = = = = = = = = =
		_addLight( new PointLight( "light1", Vec3f( 0.f, 5.f, 0.f ), WHITE, 40.f ) );
		//_addLight(
		//	new QuadLight("quadLight1", Vec3f( 1.f, 5.f, -2.f ), Vec3f( -2.f, 0.f, 0.f ), Vec3f( 0.f, 1.f, 2.f ),
		// WHITE, 40.f ) );
	}

	void Scene::_tp6()
	{
		//_addMaterial( new ColorMaterial( "GreyColor", GREY ) );
		//_addMaterial( new ColorMaterial( "RedColor", RED ) );
		_addMaterial( new LambertMaterial( "GreyLambert", GREY ) );
		_addMaterial( new LambertMaterial( "RedLambert", RED ) );
		_addMaterial( new PlasticMaterial( "GreyPlastic", GREY, 0.7f, 8.f ) ); // Color, %diffuse, shininess
		_addMaterial( new PlasticMaterial( "RedPlastic", RED, 0.7f, 8.f ) );   // Color, %diffuse, shininess
		_addMaterial( new MetalMaterial( "GoldenMetal", Vec3f( 1.f, 0.85f, 0.57f ), 0.3f, 0.3f ) );
		_addMaterial( new EmissiveMaterial( "RedEmissive", RED, 10.f ) ); // Color, intensity

		OrientationMode mode = OrientationMode::Euler;
		_addObject( new ImplicitTorus( "Torus1",
									   Vec3f( 0.f, 0.f, 3.f ),
									   Vec2f( 0.5f, 0.1f ),
									   Vec3f( glm::radians( 90.f ), glm::radians( 0.f ), glm::radians( 0.f ) ),
									   mode ) );
		_attachMaterialToObject( "RedEmissive", "Torus1" );

		_addObject( new Plane( "PlaneGround", Vec3f( 0.f, -2.f, 0.f ), Vec3f( 0.f, 1.f, 0.f ) ) );
		_attachMaterialToObject( "RedLambert", "PlaneGround" );

		_addLight( new PointLight( "light1", Vec3f( 0.f, 0.f, 1.f ), WHITE, 60.f ) );
	}

	void Scene::_killTheBunny()
	{
		_addMaterial( new ColorMaterial( "GreenColor", GREEN ) );
		_addMaterial( new MetalMaterial( "GoldenMetal", Vec3f( 1.f, 0.85f, 0.57f ), 0.3f, 0.3f ) );
		_addMaterial( new MetalMaterial( "GreenMetal", Vec3f( 0.f, 1.f, 0.f ), 0.3f, 0.3f ) );
		_addMaterial( new MetalMaterial( "GreyMetal", Vec3f( 0.7f, 0.7f, 0.7f ), 0.3f, 0.3f ) );
		_addMaterial( new MirrorMaterial( "Mirror" ) );
		_addMaterial( new EmissiveMaterial( "RedEmissive", RED, 10.f ) );

		loadFileTriangleMesh( "bunny", "data/bunny/Bunny.obj" );
		_attachMaterialToObject( "GoldenMetal", "bunny_defaultobject" );

		OrientationMode modetorus = OrientationMode::LookAtThenEuler;
		_addObject( new ImplicitTorus( "Torus1",
									   Vec3f( -6.2f, 2.2f, 3.5f ),
									   Vec2f( 0.2f, 0.21f ),
									   Vec3f( 0.f, 0.f, -6.f ),
									   modetorus,
									   Vec3f( 0.f, 90.f, 0.f ) ) );
		_attachMaterialToObject( "GreenMetal", "Torus1" );

		_addObject( new ImplicitTorus( "Torus2",
									   Vec3f( -4.96f, 1.76f, 2.8f ),
									   Vec2f( 0.2f, 0.21f ),
									   Vec3f( 0.f, 0.f, -6.f ),
									   modetorus,
									   Vec3f( 0.f, 90.f, 0.f ) ) );
		_attachMaterialToObject( "GreenMetal", "Torus2" );

		_addObject( new ImplicitTorus( "Torus3",
									   Vec3f( -3.72f, 1.32f, 2.1f ),
									   Vec2f( 0.2f, 0.21f ),
									   Vec3f( 0.f, 0.f, -6.f ),
									   modetorus,
									   Vec3f( 0.f, 90.f, 0.f ) ) );
		_attachMaterialToObject( "GreenMetal", "Torus3" );

		_addObject( new ImplicitTorus( "Torus4",
									   Vec3f( -2.48f, 0.88f, 1.4f ),
									   Vec2f( 0.2f, 0.21f ),
									   Vec3f( 0.f, 0.f, -6.f ),
									   modetorus,
									   Vec3f( 0.f, 90.f, 0.f ) ) );
		_attachMaterialToObject( "GreenMetal", "Torus4" );

		_addObject( new ImplicitTorus( "Torus5",
									   Vec3f( -1.24f, 0.44f, 0.7f ),
									   Vec2f( 0.2f, 0.21f ),
									   Vec3f( 0.f, 0.f, -6.f ),
									   modetorus,
									   Vec3f( 0.f, 90.f, 0.f ) ) );
		_attachMaterialToObject( "GreenMetal", "Torus5" );

		OrientationMode mode = OrientationMode::LookAtThenEuler;
		_addObject( new ImplicitDeathStar( "DeathStar1",
										   Vec3f( -8.f, 3.f, 5.f ),
										   1.5f,
										   1.5f,
										   2.3f,
										   Vec3f( 0.f, 0.f, -6.f ),
										   mode,
										   Vec3f( 0.f, -90.f, 0.f ) ) ); // ratio 1:0.3:0.5
		_attachMaterialToObject( "GreyMetal", "DeathStar1" );

		// Creating the lasers to shoot the bunny
		_addObject( new ImplicitLine( "Line1", Vec3f( -8.f, 3.9f, 5.f ), Vec3f( -6.2f, 2.2f, 3.5f ), 0.1f ) );
		_attachMaterialToObject( "GreenColor", "Line1" );

		_addObject( new ImplicitLine( "Line2", Vec3f( -6.7f, 2.7f, 5.f ), Vec3f( -6.2f, 2.2f, 3.5f ), 0.1f ) );
		_attachMaterialToObject( "GreenColor", "Line2" );

		_addObject( new ImplicitLine( "Line3", Vec3f( -9.4f, 2.2f, 5.f ), Vec3f( -6.2f, 2.2f, 3.5f ), 0.1f ) );
		_attachMaterialToObject( "GreenColor", "Line3" );

		_addObject( new ImplicitLine( "Line4", Vec3f( -6.2f, 2.2f, 3.5f ), Vec3f( 0.f, 0.f, 0.f ), 0.1f ) );
		_attachMaterialToObject( "GreenColor", "Line4" );

		_addObject( new Plane( "PlaneLeft", Vec3f( 5.f, 0.f, 0.f ), Vec3f( 1.f, 0.f, 0.f ) ) );
		_attachMaterialToObject( "Mirror", "PlaneLeft" );

		_addObject( new Plane( "PlaneBack", Vec3f( 0.f, 0.f, 5.f ), Vec3f( 0.f, 0.f, -1.f ) ) );
		_attachMaterialToObject( "Mirror", "PlaneBack" );

		// Lights that will illuminate one side of the bunny each
		_addLight( new PointLight( "light1", Vec3f( 0.f, 3.f, -5.f ), WHITE, 100.f ) );
		_addLight( new PointLight( "light4", Vec3f( 4.f, 0.f, 5.f ), RED, 50.f ) );
		_addLight( new PointLight( "light5", Vec3f( 0.f, 0.f, 4.f ), CYAN, 40.f ) );

		// Light above the death star to make it shine
		_addLight( new PointLight( "light3", Vec3f( -8.f, 6.f, 5.f ), WHITE, 15.f ) );

		// Light to imitate the laser light on the star
		_addLight( new PointLight( "light2", Vec3f( -6.5f, 2.4f, 3.9f ), GREEN, 15.f ) );

		// Add cylinder lights to light up the lasers.
		_addLight( new CylinderLight( "cylinderLight1",
									  Vec3f( -6.138f, 2.178f, 3.465f ),
									  7.45f,
									  0.12f,
									  Vec3f( 6.2f, -2.2f, -3.5f ),
									  GREEN,
									  3.f,
									  true ) );

		_addLight( new CylinderLight( "cylinderLight3",
									  Vec3f( -7.982f, 3.883f, 4.985f ),
									  2.89f,
									  0.12f,
									  Vec3f( 1.8f, -1.7f, -1.5f ),
									  GREEN,
									  3.f,
									  true ) );

		_addLight( new CylinderLight( "cylinderLight3",
									  Vec3f( -6.695f, 2.695f, 4.985f ),
									  1.66f,
									  0.12f,
									  Vec3f( 0.5f, -0.5f, -1.5f ),
									  GREEN,
									  3.f,
									  true ) );

		_addLight( new CylinderLight( "cylinderLight2",
									  Vec3f( -9.17f, 2.2f, 4.895f ),
									  3.53f,
									  0.12f,
									  Vec3f( 3.2f, 0.0f, -1.5f ),
									  GREEN,
									  3.f,
									  true ) );

		// Bunny brillant
		/* BaseObject * basePtr = _objectMap[ "bunny_defaultobject" ];
		if (basePtr != nullptr )
		{
			MeshTriangle * bunnyMesh = dynamic_cast<MeshTriangle *>( basePtr );
			BaseMaterial * bunnyMaterial = bunnyMesh->getMaterial();
			if (bunnyMaterial != nullptr) {
				EmissiveMaterial * emissiveMaterial = dynamic_cast<EmissiveMaterial *>( bunnyMaterial );
				_addLight( new AreaLight( "AreaLight1", bunnyMesh, emissiveMaterial ) );
				printf( "AreaLight1 created\n" );
			}
		}*/
	}

	void Scene::_DOF()
	{
		_addMaterial( new ColorMaterial( "RedColor", RED ) );
		_addMaterial( new ColorMaterial( "GreenColor", GREEN ) );
		_addMaterial( new ColorMaterial( "WhiteColor", WHITE ) );

		_addObject( new Plane( "PlaneGround", Vec3f( 0.f, -3.f, 0.f ), Vec3f( 0.f, 1.f, 0.f ) ) );
		_attachMaterialToObject( "WhiteColor", "PlaneGround" );
		_addObject( new Plane( "PlaneLeft", Vec3f( 8.f, 0.f, 0.f ), Vec3f( -1.f, 0.f, 0.f ) ) );
		_attachMaterialToObject( "WhiteColor", "PlaneLeft" );
		_addObject( new Plane( "PlaneCeiling", Vec3f( 0.f, 7.f, 0.f ), Vec3f( 0.f, -1.f, 0.f ) ) );
		_attachMaterialToObject( "WhiteColor", "PlaneCeiling" );
		_addObject( new Plane( "PlaneRight", Vec3f( -8.f, 0.f, 0.f ), Vec3f( 1.f, 0.f, 0.f ) ) );
		_attachMaterialToObject( "WhiteColor", "PlaneRight" );
		_addObject( new Plane( "PlaneFront", Vec3f( 0.f, 0.f, 20.f ), Vec3f( 0.f, 0.f, -1.f ) ) );
		_attachMaterialToObject( "WhiteColor", "PlaneFront" );
		_addObject( new Plane( "PlaneRear", Vec3f( 0.f, 0.f, -10.f ), Vec3f( 0.f, 0.f, 1.f ) ) );
		_attachMaterialToObject( "WhiteColor", "PlaneRear" );

		_addObject( new ImplicitSphere( "Sphere1", Vec3f( 0.f, 0.f, 3.f ), 1.5f ) );
		_attachMaterialToObject( "RedColor", "Sphere1" );

		_addObject( new ImplicitSphere( "Sphere2", Vec3f( 4.f, 0.f, 15.f ), 1.5f ) );
		_attachMaterialToObject( "GreenColor", "Sphere2" );

		_addLight( new PointLight( "light1", Vec3f( 0.f, 3.f, 0.f ), WHITE, 100.f ) );
	}

	void Scene::_sphere(){
		_addLight( new DirectionnalLight( "light1", Vec3f( 0.f, -1.f, 0.f ), WHITE, 30.f ) );
		_addMaterial( new ColorMaterial( "Ground", Vec3f( 0.5f, 0.5f, 0.5f ) ) );
		_addMaterial( new ColorMaterial( "Ground2", Vec3f( 1.f, 0.f, 0.f ) ) );
		_addObject( new Sphere( "Sphere", Vec3f(0.f,1.f,0.f), 1.f ) );
		_attachMaterialToObject( "Ground", "Sphere" );
		_addObject(new Plane("Plane", Vec3f(0,0,0), Vec3f(0,1,0)));
		_attachMaterialToObject("Ground2", "Plane");

	}
	
	void Scene::_spheres() 
	{
		_addLight( new DirectionnalLight( "light1", Vec3f( 0.f, -1.f, 0.f ), WHITE, 1.f ) );
		
		_addMaterial( new ColorMaterial( "Ground", Vec3f( 0.5f, 0.5f, 0.5f ) ) );
		_addMaterial( new MirrorMaterial( "Mirror" ) );
		_addMaterial( new TransparentMaterial( "Transparent" ) );

		_addObject( new Plane( "Ground", Vec3f( 0.f, 0.f, 0.f ), Vec3f( 0.f, 1.f, 0.f ) ) );
		_attachMaterialToObject( "Ground", "Ground" );
		for ( int i = -11; i < 11; i++ )
		{
			for (int j = -11; j < 11; j++) {
				double choose_mat = randomDouble();

				Vec3f center = Vec3f( i + 0.9 * randomDouble(), 0.2, j + 0.9 * randomDouble() );

				if ( (center - Vec3f(4, 0.2, 0)).length() > 0.9 ) 
				{
					if ( choose_mat < 0.6 ) // diffuse
					{
						std::string nameTemp = std::to_string( i ) + "_" + std::to_string( j );
						_addMaterial( new ColorMaterial( "Color" + nameTemp ) );
						//_addObject( new Sphere( "Sphere" + nameTemp, center, center + Vec3f(0.0,0.3,0.0), 0.2 ) );
						_addObject( new Sphere( "Sphere" + nameTemp, center, 0.2 ) );
						_attachMaterialToObject( "Color" + nameTemp, "Sphere" + nameTemp );
					}
					else if ( choose_mat < 0.8 ) // metal
					{
						std::string nameTemp = std::to_string( i ) + "_" + std::to_string( j );
						_addMaterial( new MetalMaterial( "Metal" + std::to_string(i) + "_" + std::to_string(j)) );
						_addObject( new Sphere( "Sphere" + nameTemp, center, 0.2 ) );
						_attachMaterialToObject( "Metal" + nameTemp, "Sphere" + nameTemp );
					}
					else if ( choose_mat < 0.88 ) 
					{
						std::string nameTemp = std::to_string( i ) + "_" + std::to_string( j );
						_addObject( new Sphere( "Sphere" + nameTemp, center, 0.2 ) );
						_attachMaterialToObject( "Mirror", "Sphere" + nameTemp );
					} 
					else
					{
						std::string nameTemp = std::to_string( i ) + "_" + std::to_string( j );
						_addObject( new Sphere( "Sphere" + nameTemp, center, 0.2 ) );
						_attachMaterialToObject( "Transparent", "Sphere" + nameTemp );
					}
				}
			}
		}

		_addObject( new Sphere( "Sphere1", Vec3f(0,1,0), 1.f ) );
		_attachMaterialToObject( "Transparent", "Sphere1" );
		_addObject( new Sphere( "Sphere2", Vec3f( -4, 1, 0 ), 1.f ) );
		_attachMaterialToObject( "Ground", "Sphere2" );
		_addObject( new Sphere( "Sphere3", Vec3f( 4, 1, 0 ), 1.f ) );
		_attachMaterialToObject( "Mirror", "Sphere3" );

	}

} // namespace RT
