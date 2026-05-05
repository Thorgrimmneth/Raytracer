#include "lights/light.cuh"
#include "scene.cuh"

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