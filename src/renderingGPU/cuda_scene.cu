#include "lights/cuda_light.cuh"
#include "cuda_scene.cuh"
#include "../defines.hpp"
#include "../scene.hpp"
#include "../objects/base_object.hpp"
#include "../objects/plane.hpp"
#include "../objects/sphere.hpp"
#include "../geometry/sphere_geometry.hpp"
#include "../geometry/plane_geometry.hpp"
#include "../objects/triangle_mesh.hpp"
#include "../materials/base_material.hpp"
#include "../materials/color_material.hpp"
#include "../materials/emissive_material.hpp"
#include "../materials/lambert_material.hpp"
#include "../materials/metal_material.hpp"
#include "../materials/mirror_material.hpp"
#include "../materials/plastic_material.hpp"
#include "../materials/transparent_material.hpp"
#include "../lights/point_light.hpp"
#include "../lights/quad_light.hpp"
#include "../lights/cylinder_light.hpp"
#include "../lights/directionnal_light.hpp"

__device__ bool CudaScene::intersect(const Ray &p_ray, const float p_tMin, const float p_tMax, HitRecord &p_hitRecord) const
{
	float tMax = p_tMax;
	bool hit = false;
	if (bvhScene.intersect(p_ray, p_tMin, tMax, p_hitRecord))
	{
		tMax = p_hitRecord.distance; // update tMax to conserve the nearest hit
		hit = true;
	}
	return hit;
}

__device__ bool CudaScene::intersectAny(const Ray &p_ray, const float p_tMin, const float p_tMax) const
{
	return bvhScene.intersectAny(p_ray, p_tMin, p_tMax, materials);
}

AABB convertBBOX(RT::AABB bbox)
{
	AABB nbbox;
	nbbox.min = make_float4(bbox.getMin().x, bbox.getMin().y, bbox.getMin().z, 0.f);
	nbbox.max = make_float4(bbox.getMax().x, bbox.getMax().y, bbox.getMax().z, 0.f);
	return nbbox;
}

int flattenBVH(const RT::BVHNode *node,
			   std::vector<BVH> &outNodes)
{
	if (!node)
		return -1;
	int index = outNodes.size();
	outNodes.push_back({});

	BVH gpuNode;
	gpuNode.bbox = convertBBOX(node->_aabb);

	if (node->isLeaf())
	{
		gpuNode.left = -1;
		gpuNode.right = -1;
		gpuNode.firstTriangleIndex = node->_firstTriangleId;
		gpuNode.lastTriangleIndex = node->_lastTriangleId;
	}
	else
	{
		gpuNode.firstTriangleIndex = -1;
		gpuNode.lastTriangleIndex = -1;

		gpuNode.left = flattenBVH(node->_left, outNodes);
		gpuNode.right = flattenBVH(node->_right, outNodes);
	}

	outNodes[index] = gpuNode;
	return index;
}

Material convertMaterial(RT::BaseMaterial *bm)
{
	Material m;
	if (bm->getType() == RT::MaterialType::EMISSIVE)
	{
		RT::EmissiveMaterial *material = dynamic_cast<RT::EmissiveMaterial *>(bm);
		m.color = make_float3(material->getFlatColor().x, material->getFlatColor().y, material->getFlatColor().z);
		m.intensity = material->getIntensity();
		m.type = MaterialType::EMISSIVE;
	}
	else if (bm->getType() == RT::MaterialType::LAMBERT)
	{
		RT::LambertMaterial *material = dynamic_cast<RT::LambertMaterial *>(bm);
		m.color = make_float3(material->getFlatColor().x, material->getFlatColor().y, material->getFlatColor().z);
		m.type = MaterialType::LAMBERT;
	}
	else if (bm->getType() == RT::MaterialType::METAL)
	{
		RT::MetalMaterial *material = dynamic_cast<RT::MetalMaterial *>(bm);
		m.color = make_float3(material->getFlatColor().x, material->getFlatColor().y, material->getFlatColor().z);
		m.metalness = material->getMetalness();
		m.ruggedness = material->getRuggedness();
		m.alpha = m.ruggedness * m.ruggedness;
		m.type = MaterialType::METAL;
	}
	else if (bm->getType() == RT::MaterialType::MIRROR)
	{
		RT::MirrorMaterial *material = dynamic_cast<RT::MirrorMaterial *>(bm);
		m.color = make_float3(material->getFlatColor().x, material->getFlatColor().y, material->getFlatColor().z);
		m.type = MaterialType::MIRROR;
	}
	else if (bm->getType() == RT::MaterialType::PLASTIC)
	{
		RT::PlasticMaterial *material = dynamic_cast<RT::PlasticMaterial *>(bm);
		m.color = make_float3(material->getFlatColor().x, material->getFlatColor().y, material->getFlatColor().z);
		m.shininess = material->getShininess();
		m.type = MaterialType::PLASTIC;
	}
	else if (bm->getType() == RT::MaterialType::TRANSPARENT)
	{
		RT::TransparentMaterial *material = dynamic_cast<RT::TransparentMaterial *>(bm);
		m.color = make_float3(material->getFlatColor().x, material->getFlatColor().y, material->getFlatColor().z);
		m.ior = material->getIOR();
		m.type = MaterialType::TRANSPARENT;
	}
	return m;
}

__host__
void CudaScene::uploadObjects(
					   std::vector<Sphere> spheresGPU,
					   std::vector<Plane> planesGPU,
					   std::vector<TriangleMesh> triangleMeshesGPU,
					   std::vector<float3> verticesGPU,
					   std::vector<BaseObject> primitivesGPU)
{
	cudaMalloc(&primitives,
			   primitivesGPU.size() * sizeof(BaseObject));

	cudaMemcpy(primitives,
			   primitivesGPU.data(),
			   primitivesGPU.size() * sizeof(BaseObject),
			   cudaMemcpyHostToDevice);

	bvhScene = BVHScene::buildBVHScene(&primitivesGPU, &spheresGPU, &planesGPU, &triangleMeshesGPU);

	nbSpheres = spheresGPU.size();

	nbPlanes = planesGPU.size();

	cudaMalloc(&spheres,
			   spheresGPU.size() * sizeof(Sphere));

	cudaMemcpy(spheres,
			   spheresGPU.data(),
			   spheresGPU.size() * sizeof(Sphere),
			   cudaMemcpyHostToDevice);

	cudaMalloc(&planes,
			   planesGPU.size() * sizeof(Plane));

	cudaMemcpy(planes,
			   planesGPU.data(),
			   planesGPU.size() * sizeof(Plane),
			   cudaMemcpyHostToDevice);

	nbTriangleMeshes = triangleMeshesGPU.size();

	cudaMalloc(&triangleMeshes,
			   triangleMeshesGPU.size() * sizeof(TriangleMesh));

	cudaMemcpy(triangleMeshes,
			   triangleMeshesGPU.data(),
			   triangleMeshesGPU.size() * sizeof(TriangleMesh),
			   cudaMemcpyHostToDevice);

	bvhScene.d_spheres = spheres;
	bvhScene.d_planes = planes;
	bvhScene.d_meshes = triangleMeshes;
	
}

__host__
void CudaScene::uploadLights(
                             std::vector<Light> lightsGPU){
	nbLights = lightsGPU.size();
	cudaMalloc(&lights,
			   lightsGPU.size() * sizeof(Light));

	cudaMemcpy(lights,
			   lightsGPU.data(),
			   lightsGPU.size() * sizeof(Light),
			   cudaMemcpyHostToDevice);
}

__host__ 
void CudaScene::uploadMaterials(std::vector<Material> materialsGPU)
{
	nbMaterials = materialsGPU.size();

	if (nbMaterials > 0)
	{
		cudaMalloc(&materials,
				   nbMaterials * sizeof(Material));

		cudaMemcpy(materials,
				   materialsGPU.data(),
				   nbMaterials * sizeof(Material),
				   cudaMemcpyHostToDevice);
	}
	else
	{
		materials = nullptr;
	}
}
CudaScene uploadSceneToGPU(const RT::Scene &scene, float4 sunDir)
{
	CudaScene gpuScene;

	std::vector<Sphere> spheresGPU;
	std::vector<Plane> planesGPU;
	std::vector<TriangleMesh> triangleMeshesGPU;
	std::vector<float3> verticesGPU;
	std::vector<BaseObject> primitivesGPU;
	std::vector<Material> materialsGPU;
	std::vector<Light> lightsGPU;

	for (const auto &pair : scene.getObject())
	{
		RT::BaseObject *obj = pair.second;

		if (obj->getType() == RT::ObjectType::Sphere)
		{
			RT::Sphere *sphereTemp = dynamic_cast<RT::Sphere *>(obj);
			if (!sphereTemp)
			{
				std::cout << "Dynamic cast failed for Sphere\n";
				continue;
			}
			const RT::SphereGeometry &sphere = sphereTemp->getGeometry();

			Sphere s;
			RT::Ray center = sphere.getCenters();
			s.center1 = make_float3(center.getOrigin().x, center.getOrigin().y, center.getOrigin().z);
			s.center2 = make_float3(center.pointAtT(1.f).x, center.pointAtT(1.f).y, center.pointAtT(1.f).z); // center2
			s.radius = sphere.getRadius();

			float3 r = make_float3(s.radius, s.radius, s.radius);

			AABB box1;
			box1.min = toFloat4(s.center1 - r);
			box1.max = toFloat4(s.center1 + r);

			AABB box2;
			box2.min = toFloat4(s.center2 - r);
			box2.max = toFloat4(s.center2 + r);

			box1.min = getMin(box1.min, box2.min);
			box1.max = getMax(box1.max, box2.max);
			primitivesGPU.push_back(BaseObject{box1, ObjectType::SPHERE, (int)spheresGPU.size()});
			Material mat = convertMaterial(obj->getMaterial());
			materialsGPU.push_back(mat);
			s.materialIndex = materialsGPU.size() - 1;
			spheresGPU.push_back(s);
			if (mat.type == MaterialType::EMISSIVE)
			{
				Light l;
				l.type = LightType::SPHERE_GEOM;
				l.geomIndex = spheresGPU.size() - 1;
				l.area = 4.f * M_PI * s.radius * s.radius;
				lightsGPU.push_back(l);
			}
		}
		else if (obj->getType() == RT::ObjectType::Plane)
		{
			RT::Plane *planeTemp = dynamic_cast<RT::Plane *>(obj);
			if (!planeTemp)
			{
				std::cout << "Dynamic cast failed for Plane\n";
				continue;
			}
			const RT::PlaneGeometry &plane = planeTemp->getGeometry();

			Plane p;
			p.delta = plane.getDelta();
			p.normal = make_float3(plane.getNormal().x, plane.getNormal().y, plane.getNormal().z);
			materialsGPU.push_back(convertMaterial(obj->getMaterial()));
			p.materialIndex = materialsGPU.size() - 1;
			AABB bbox;
			bbox.min = make_float4(-1e3f, plane.getPosition().y, -1e3f, 0.f);
			bbox.max = make_float4(1e3f, plane.getPosition().y, 1e3f, 0.f);
			primitivesGPU.push_back(BaseObject{bbox, ObjectType::PLANE, (int)planesGPU.size()});
			planesGPU.push_back(p);
		}
		else if (obj->getType() == RT::ObjectType::TriangleMesh)
		{
			RT::MeshTriangle *meshCPU = dynamic_cast<RT::MeshTriangle *>(obj);
			if (!meshCPU)
			{
				std::cout << "Dynamic cast failed for TriangleMesh\n";
				continue;
			}
			TriangleMesh tm{};

			// ===== Vertices =====
			std::vector<float3> vertices;
			for (const RT::Vec3f &v : meshCPU->getVertices())
				vertices.push_back(make_float3(v.x, v.y, v.z));

			tm.vertexCount = vertices.size();

			cudaMalloc(&tm.vertices, vertices.size() * sizeof(float3));
			cudaMemcpy(tm.vertices, vertices.data(),
					   vertices.size() * sizeof(float3),
					   cudaMemcpyHostToDevice);

			// ===== Normals =====
			std::vector<float3> normals;
			for (const RT::Vec3f &n : meshCPU->getNormals())
				normals.push_back(make_float3(n.x, n.y, n.z));

			cudaMalloc(&tm.normals, normals.size() * sizeof(float3));
			cudaMemcpy(tm.normals, normals.data(),
					   normals.size() * sizeof(float3),
					   cudaMemcpyHostToDevice);

			// ===== UV =====
			std::vector<float2> uvs;
			for (const RT::Vec2f &uv : meshCPU->getUVS())
				uvs.push_back(make_float2(uv.x, uv.y));

			cudaMalloc(&tm.uvs, uvs.size() * sizeof(float2));
			cudaMemcpy(tm.uvs, uvs.data(),
					   uvs.size() * sizeof(float2),
					   cudaMemcpyHostToDevice);

			// ===== Triangles =====
			std::vector<TriangleMeshGeometry> triangles;
            std::vector<float> triangleAreaCdf;
            float meshArea = 0.f;
            for (const auto &tri : meshCPU->getTriangles())
            {
                TriangleMeshGeometry t;
                t.i0 = tri.getV0();
                t.i1 = tri.getV1();
                t.i2 = tri.getV2();
                triangles.push_back(t);

                const float3 &v0 = vertices[t.i0];
                const float3 &v1 = vertices[t.i1];
                const float3 &v2 = vertices[t.i2];
                float area = 0.5f * length(cross(v1 - v0, v2 - v0));
                meshArea += area;
                triangleAreaCdf.push_back(meshArea);
            }

            tm.triangleCount = triangles.size();
            tm.meshArea = meshArea;

            cudaMalloc(&tm.triangles, triangles.size() * sizeof(TriangleMeshGeometry));
            cudaMemcpy(tm.triangles, triangles.data(),
                       triangles.size() * sizeof(TriangleMeshGeometry),
                       cudaMemcpyHostToDevice);

            cudaMalloc(&tm.triangleAreaCdf, triangleAreaCdf.size() * sizeof(float));
            cudaMemcpy(tm.triangleAreaCdf, triangleAreaCdf.data(),
                       triangleAreaCdf.size() * sizeof(float),
                       cudaMemcpyHostToDevice);

            // ===== BVH =====
            std::vector<BVH> linearBVH;
            flattenBVH(meshCPU->getBVH().getRoot(), linearBVH);

            tm.bvhNodeCount = linearBVH.size();

            cudaMalloc(&tm.bvhNodes, linearBVH.size() * sizeof(BVH));
            cudaMemcpy(tm.bvhNodes, linearBVH.data(),
                       linearBVH.size() * sizeof(BVH),
                       cudaMemcpyHostToDevice);

			Material mat = convertMaterial(obj->getMaterial());
			materialsGPU.push_back(mat);
			tm.materialIndex = materialsGPU.size() - 1;
			primitivesGPU.push_back(BaseObject{linearBVH[0].bbox, ObjectType::TRIANGLE, (int)triangleMeshesGPU.size()});
			triangleMeshesGPU.push_back(tm);

			if (mat.type == MaterialType::EMISSIVE)
			{
				float meshArea = 0.f;
				for (const auto &tri : triangles)
				{
					const float3 &v0 = vertices[tri.i0];
					const float3 &v1 = vertices[tri.i1];
					const float3 &v2 = vertices[tri.i2];
					meshArea += 0.5f * length(cross(v1 - v0, v2 - v0));
				}
				Light l;
				l.type = LightType::MESH_GEOM;
				l.geomIndex = triangleMeshesGPU.size() - 1;
				l.area = meshArea;
				lightsGPU.push_back(l);
			}
		}
	}

	for (const auto &light : scene.getLights())
	{
		if (light->getType() == RT::LightType::POINT)
		{
			RT::PointLight *pointLight = dynamic_cast<RT::PointLight *>(light);
			Light l;
			l.position = make_float3(pointLight->getPosition().x, pointLight->getPosition().y, pointLight->getPosition().z);
			l.color = make_float3(pointLight->getFlatColor().x, pointLight->getFlatColor().y, pointLight->getFlatColor().z);
			l.power = pointLight->getPower();
			l.type = LightType::POINT;
			lightsGPU.push_back(l);
		}
		else if (light->getType() == RT::LightType::QUAD)
		{
			RT::QuadLight *quadLight = dynamic_cast<RT::QuadLight *>(light);
			Light l;
			l.position = make_float3(quadLight->getPosition().x, quadLight->getPosition().y, quadLight->getPosition().z);
			l.color = make_float3(quadLight->getFlatColor().x, quadLight->getFlatColor().y, quadLight->getFlatColor().z);
			l.area = quadLight->getArea();
			l.normal = make_float3(quadLight->getNormal().x, quadLight->getNormal().y, quadLight->getNormal().z);
			l.u = make_float3(quadLight->getU().x, quadLight->getU().y, quadLight->getU().z);
			l.v = make_float3(quadLight->getV().x, quadLight->getV().y, quadLight->getV().z);
			l.power = quadLight->getPower();
			l.type = LightType::QUAD;
			lightsGPU.push_back(l);
		}
		else if (light->getType() == RT::LightType::CYLINDER)
		{
			RT::CylinderLight *cylinderLight = dynamic_cast<RT::CylinderLight *>(light);
			Light l;
			l.position = make_float3(cylinderLight->getPosition().x, cylinderLight->getPosition().y, cylinderLight->getPosition().z);
			l.color = make_float3(cylinderLight->getFlatColor().x, cylinderLight->getFlatColor().y, cylinderLight->getFlatColor().z);
			l.power = cylinderLight->getPower();
			l.height = cylinderLight->getHeight();
			l.radius = cylinderLight->getRayon();
			l.direction = make_float3(cylinderLight->getDirection().x, cylinderLight->getDirection().y, cylinderLight->getDirection().z);
			l.area = cylinderLight->getArea();
			l.type = LightType::CYLINDER;
			lightsGPU.push_back(l);
		}
		else if (light->getType() == RT::LightType::DIRECTIONNAL)
		{
			RT::DirectionnalLight *directionnalLight = dynamic_cast<RT::DirectionnalLight *>(light);
			Light l;
			l.color = make_float3(directionnalLight->getFlatColor().x, directionnalLight->getFlatColor().y, directionnalLight->getFlatColor().z);
			l.power = directionnalLight->getPower();
			l.direction = make_float3(directionnalLight->getDirection().x, directionnalLight->getDirection().y, directionnalLight->getDirection().z);
			l.type = LightType::DIRECTIONNAL;
			lightsGPU.push_back(l);
		}
	}
	Light l;
	l.color = make_float3(1.f);
	l.power = 1.f;
	l.area = 1.f;
	l.direction = toFloat3(sunDir);
	l.type = LightType::SUN;
	lightsGPU.push_back(l);
	
	gpuScene.uploadObjects(spheresGPU, planesGPU, triangleMeshesGPU, verticesGPU, primitivesGPU);
	gpuScene.uploadLights(lightsGPU);
	gpuScene.uploadMaterials(materialsGPU);

	// printf("nb objects bvh %i\n", gpuScene.bvhScene.nbObjects);
	// printf("nb nodes bvh %i\n", gpuScene.bvhScene.nbNodes);
	return gpuScene;
}

CudaScene spheresScene(float4 sunDir)
{
    CudaScene gpuScene;

    std::vector<Sphere> spheresGPU;
    std::vector<Plane> planesGPU;
    std::vector<TriangleMesh> triangleMeshesGPU;
    std::vector<float3> verticesGPU;
    std::vector<BaseObject> primitivesGPU;
    std::vector<Material> materialsGPU;
    std::vector<Light> lightsGPU;

    // ===== PLAN =====
    {
        Plane p;
        p.delta = 0.f;
        p.normal = make_float3(0.f, 1.f, 0.f);

        Material ground = Material(make_float3(0.5f));
        materialsGPU.push_back(ground);
        p.materialIndex = materialsGPU.size() - 1;

        AABB bbox;
        bbox.min = make_float4(-1e3f, 0.f, -1e3f, 0.f);
        bbox.max = make_float4(1e3f, 0.f, 1e3f, 0.f);

        primitivesGPU.push_back(BaseObject{bbox, ObjectType::PLANE, (int)planesGPU.size()});
        planesGPU.push_back(p);
    }

    // ===== MATERIALS DE BASE =====
    Material mirror = Material(make_float3(0.f), true);
    Material transparent = Material(make_float3(0.f), 1.5f, MaterialType::TRANSPARENT);
    Material emissive = Material(make_float3(1.f, 0.f, 0.f), 11.f, MaterialType::EMISSIVE);

    int mirrorIdx = materialsGPU.size(); materialsGPU.push_back(mirror);
    int transparentIdx = materialsGPU.size(); materialsGPU.push_back(transparent);
    int emissiveIdx = materialsGPU.size(); materialsGPU.push_back(emissive);

    float bigRadius = 1.0f;
    float smallRadius = 0.2f;
    float margin = 0.05f;
    float minDist = bigRadius + smallRadius + margin;

    // ===== PETITES SPHERES =====
    for (int i = -11; i < 11; i++)
    {
        for (int j = -11; j < 11; j++)
        {
            double choose_mat = RT::randomDouble();

            float3 center = make_float3(
                i + 0.9f * RT::randomFloat(),
                0.2f,
                j + 0.9f * RT::randomFloat()
            );

            if (
                length(center - make_float3(4.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(0.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(-4.f, 1.f, 0.f)) < minDist)
                continue;

            Sphere s;
            s.center1 = center;
            s.center2 = center;
            s.radius = 0.2f;

            // ===== MATERIAL =====
            if (choose_mat < 0.6)
            {
                Material mat = Material(make_float3(
                    RT::randomFloat(),
                    RT::randomFloat(),
                    RT::randomFloat()
                ));
                materialsGPU.push_back(mat);
                s.materialIndex = materialsGPU.size() - 1;
            }
            else if (choose_mat < 0.8)
            {
                Material mat = Material::randomMetal();
                materialsGPU.push_back(mat);
                s.materialIndex = materialsGPU.size() - 1;
            }
            else if (choose_mat < 0.88)
            {
                s.materialIndex = mirrorIdx;
            }
            else
            {
                s.materialIndex = transparentIdx;
            }

            // ===== AABB =====
            float3 r = make_float3(s.radius);
            AABB box;
            box.min = toFloat4(center - r);
            box.max = toFloat4(center + r);

            primitivesGPU.push_back(BaseObject{
                box,
                ObjectType::SPHERE,
                (int)spheresGPU.size()
            });

            spheresGPU.push_back(s);
        }
    }

    // ===== GROSSES SPHERES =====
    auto addBigSphere = [&](float3 center, float radius, int matIndex)
    {
        Sphere s;
        s.center1 = center;
        s.center2 = center;
        s.radius = radius;
        s.materialIndex = matIndex;

        float3 r = make_float3(radius);
        AABB box;
        box.min = toFloat4(center - r);
        box.max = toFloat4(center + r);

        primitivesGPU.push_back(BaseObject{
            box,
            ObjectType::SPHERE,
            (int)spheresGPU.size()
        });

        spheresGPU.push_back(s);
		Material& mat = materialsGPU[matIndex];

		if (mat.type == MaterialType::EMISSIVE)
		{
			Light l;
			l.type = LightType::SPHERE_GEOM;
			l.geomIndex = spheresGPU.size() - 1;
			l.area = 4.f * M_PI * radius * radius;
			lightsGPU.push_back(l);
		}
    };

    addBigSphere(make_float3(0.f, 1.f, 0.f), 1.f, transparentIdx);
    addBigSphere(make_float3(-4.f, 1.f, 0.f), 1.f, emissiveIdx);
    addBigSphere(make_float3(4.f, 1.f, 0.f), 1.f, mirrorIdx);

    // ===== LIGHT (SUN) =====
    Light l;
    l.color = make_float3(1.f);
    l.power = 1.f;
    l.area = 1.f;
    l.direction = toFloat3(sunDir);
    l.type = LightType::SUN;
    lightsGPU.push_back(l);

    // ===== UPLOAD =====
    gpuScene.uploadObjects(spheresGPU, planesGPU, triangleMeshesGPU, verticesGPU, primitivesGPU);
    gpuScene.uploadLights(lightsGPU);
    gpuScene.uploadMaterials(materialsGPU);

    return gpuScene;
}