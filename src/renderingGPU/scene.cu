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
    // =========================
    // Upload primitives
    // =========================
    int nbObjects = primitivesGPU.size();

    cudaMalloc(&primitives, nbObjects * sizeof(BaseObject));
    cudaMemcpy(primitives,
               primitivesGPU.data(),
               nbObjects * sizeof(BaseObject),
               cudaMemcpyHostToDevice);

    // =========================
    // Build BVH
    // =========================
    bvhScene = BVHScene::buildBVHScene(&primitivesGPU,
                                       &spheresGPU,
                                       &planesGPU,
                                       &triangleMeshesGPU);

    // =========================
    // Upload spheres
    // =========================
    nbSpheres = spheresGPU.size();

    cudaMalloc(&spheres, nbSpheres * sizeof(Sphere));
    cudaMemcpy(spheres,
               spheresGPU.data(),
               nbSpheres * sizeof(Sphere),
               cudaMemcpyHostToDevice);

    // =========================
    // Upload planes
    // =========================
    nbPlanes = planesGPU.size();

    cudaMalloc(&planes, nbPlanes * sizeof(Plane));
    cudaMemcpy(planes,
               planesGPU.data(),
               nbPlanes * sizeof(Plane),
               cudaMemcpyHostToDevice);

    // =========================
    // Upload meshes
    // =========================
    nbTriangleMeshes = triangleMeshesGPU.size();

    cudaMalloc(&triangleMeshes,
               nbTriangleMeshes * sizeof(TriangleMesh));

    cudaMemcpy(triangleMeshes,
               triangleMeshesGPU.data(),
               nbTriangleMeshes * sizeof(TriangleMesh),
               cudaMemcpyHostToDevice);

    bvhScene.d_primitives = primitives;
    bvhScene.d_spheres    = spheres;
    bvhScene.d_planes     = planes;
    bvhScene.d_meshes     = triangleMeshes;
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

        Material ground = Material::makeMaterial(make_float3(0.5f), LAMBERT, 1.0f);
        materialsGPU.push_back(ground);
        p.materialIndex = materialsGPU.size() - 1;

        primitivesGPU.push_back(BaseObject{make_float3(-1e3f, 0.f, -1e3f), make_float3(1e3f, 0.f, 1e3f),ObjectType::PLANE, (int)planesGPU.size()});
        planesGPU.push_back(p);
    }

    // ===== MATERIALS DE BASE =====
    Material mirror      = Material::makeMaterial(make_float3(0.f), MIRROR);
    Material transparent = Material::makeMaterial(make_float3(1.f), TRANSPARENT, 0.f, 0.f, 1.5f);
    Material emissive    = Material::makeMaterial(make_float3(1.f, 0.f, 0.f), EMISSIVE, 0.f, 0.f, 1.f, 11.f);

    int mirrorIdx = materialsGPU.size(); materialsGPU.push_back(mirror);
    int transparentIdx = materialsGPU.size(); materialsGPU.push_back(transparent);
    int emissiveIdx = materialsGPU.size(); materialsGPU.push_back(emissive);

    float bigRadius = 1.0f;
    float smallRadius = 0.2f;
    float margin = 0.05f;
    float minDist = bigRadius + smallRadius + margin;

    int numberOfSpheresPerSide = 11;
    // ===== PETITES SPHERES =====
    for (int i = -numberOfSpheresPerSide; i < numberOfSpheresPerSide; i++)
    {
        for (int j = -numberOfSpheresPerSide; j < numberOfSpheresPerSide; j++)
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
                Material mat = Material::makeMaterial(
                    make_float3(
                        RT::randomFloat(),
                        RT::randomFloat(),
                        RT::randomFloat()
                    ),
                    LAMBERT,
                    1.0f
                );
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

            primitivesGPU.push_back(BaseObject{
                center - r,
                center + r,
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

        primitivesGPU.push_back(BaseObject{
            center - r,
            center + r,
            ObjectType::SPHERE,
            (int)spheresGPU.size()
        });

        spheresGPU.push_back(s);
		Material& mat = materialsGPU[matIndex];

		if (mat.type() == MaterialType::EMISSIVE)
		{
			Light l;
			l.metadata = Light::packMetadata(LightType::SPHERE_GEOM, spheresGPU.size() - 1);
			lightsGPU.push_back(l);
		}
    };

    addBigSphere(make_float3(0.f, 1.f, 0.f), 1.f, transparentIdx);
    addBigSphere(make_float3(-4.f, 1.f, 0.f), 1.f, emissiveIdx);
    addBigSphere(make_float3(4.f, 1.f, 0.f), 1.f, mirrorIdx);

    // ===== LIGHT (SUN) =====
    Light l;
    l.color_power = make_float4(1.f, 0.95f, 0.9f, 100.f);
    l.direction = make_float4(sunDir.x, sunDir.y, sunDir.z, 0.f);
    l.metadata = Light::packMetadata(LightType::SUN, 0);
    lightsGPU.push_back(l);

    // ===== UPLOAD =====
    gpuScene.uploadObjects(spheresGPU, planesGPU, triangleMeshesGPU, verticesGPU, primitivesGPU);
    gpuScene.uploadLights(lightsGPU);
    gpuScene.uploadMaterials(materialsGPU);

    printf("Size of one BVH node: %zu bytes\n", sizeof(BVHSceneNode));
    printf("Size of AABB: %zu bytes\n", sizeof(AABB));
    printf("Size of BaseObject: %zu bytes\n", sizeof(BaseObject));
    printf("Size of Sphere: %zu bytes\n", sizeof(Sphere));
    printf("Size of Plane: %zu bytes\n", sizeof(Plane));
    printf("Size of Material: %zu bytes\n", sizeof(Material));
    printf("Size of Light: %zu bytes\n", sizeof(Light));
    printf("\n");
    size_t totalSize = 0;
    totalSize += gpuScene.nbSpheres * sizeof(Sphere);
    printf("Size of spheres: %zu bytes. %2.2f gain compared to v1\n", gpuScene.nbSpheres * sizeof(Sphere), (1.f - (gpuScene.nbSpheres * sizeof(Sphere) / 38240.f)) * 100.f);
    totalSize += gpuScene.nbPlanes * sizeof(Plane);
    printf("Size of planes: %zu bytes. %2.2f gain compared to v1\n", gpuScene.nbPlanes * sizeof(Plane), (1.f - (gpuScene.nbPlanes * sizeof(Plane) / 20.f))* 100.f);
    totalSize += gpuScene.nbTriangleMeshes * sizeof(TriangleMesh);
    printf("Size of triangle meshes: %zu bytes. %2.2f gain compared to v1\n", gpuScene.nbTriangleMeshes * sizeof(TriangleMesh), (1.f - (gpuScene.nbTriangleMeshes * sizeof(TriangleMesh) / 1.f)) * 100.f);
    totalSize += gpuScene.nbMaterials * sizeof(Material);
    printf("Size of materials: %zu bytes. %2.2f gain compared to v1\n", gpuScene.nbMaterials * sizeof(Material), (1.f - (gpuScene.nbMaterials * sizeof(Material) / 15360.f)) * 100.f);
    totalSize += gpuScene.nbLights * sizeof(Light);
    printf("Size of lights: %zu bytes. %2.2f gain compared to v1\n", gpuScene.nbLights * sizeof(Light), (1.f - (gpuScene.nbLights * sizeof(Light) / 192.f)) * 100.f);
    totalSize += verticesGPU.size() * sizeof(float3);
    printf("Size of vertices: %zu bytes. %2.2f gain compared to v1\n", verticesGPU.size() * sizeof(float3), (1.f - (verticesGPU.size() * sizeof(float3) / 1.f)) * 100.f);
    totalSize += primitivesGPU.size() * sizeof(BaseObject);
    printf("Size of primitives: %zu bytes. %2.2f gain compared to v1\n", primitivesGPU.size() * sizeof(BaseObject), (1.f - (primitivesGPU.size() * sizeof(BaseObject) / 22992.f)) * 100.f);
    totalSize += gpuScene.bvhScene.getDeviceSize();
    printf("BVH size: %zu bytes. %2.2f gain compared to v1\n", gpuScene.bvhScene.getDeviceSize(), (1.f - (gpuScene.bvhScene.getDeviceSize() / 68928.f)) * 100.f);
    printf("Total size of GPU data: %zu bytes. %2.2f gain compared to v1\n", totalSize, (1.f - (totalSize / 145732.f)) * 100.f);
    return gpuScene;
}

__device__
float CudaScene::lightPdf(
    const float3& origin,
    const float3& dir) const
{
    Ray ray(origin, dir);

    HitRecord hit;

    if(!intersect(ray, 1e-4f, 1e30f, hit))
        return 0.f;

    const Material& mtl =
        materials[hit.materialIndex];

    if(mtl.type() != MaterialType::EMISSIVE)
        return 0.f;

    float pdf = 0.f;

    // sphere emissive
    if(hit.objectType == HIT_SPHERE)
    {
        const Sphere& s =
            spheres[hit.objectIndex];

        float3 lightCenter = s.center1;
        float radius = s.radius;

        float dist2 =
            length2(hit.point - origin);

        float3 n =
            normalize(hit.point - lightCenter);

        float cosTheta =
            max(dot(n, -dir), 0.f);

        if(cosTheta <= 0.f)
            return 0.f;

        float area =
            4.f * GPUPIf * radius * radius;

        float pdfArea = 1.f / area;

        pdf =
            pdfArea * dist2 / cosTheta;
    }

    // triangle mesh emissive
    else if(hit.objectType == HIT_TRIANGLE_MESH)
    {
        const TriangleMesh& mesh =
            triangleMeshes[hit.objectIndex];

        float3 n = hit.normal;

        float dist2 =
            length2(hit.point - origin);

        float cosTheta =
            max(dot(n, -dir), 0.f);

        if(cosTheta <= 0.f)
            return 0.f;

        float pdfArea =
            1.f / mesh.meshArea;

        pdf =
            pdfArea * dist2 / cosTheta;
    }

    // lumière choisie uniformément
    pdf *= (1.f / nbLights);

    return pdf;
};