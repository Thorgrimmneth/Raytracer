
#include "scene.cuh"

HOST
void CudaScene::uploadObjects(
    std::vector<Sphere> spheresGPU,
    std::vector<Plane> planesGPU,
    std::vector<TriangleMesh> triangleMeshesGPU,
    std::vector<BaseObject> primitivesGPU,
    std::vector<ImplicitSphere> implicitSpheresGPU)
{
    // =========================
    // Upload primitives
    // =========================
    int nbObjects = primitivesGPU.size();

    if (nbObjects > 0)
    {
        cudaMalloc(&primitives, nbObjects * sizeof(BaseObject));
        cudaMemcpy(primitives,
                   primitivesGPU.data(),
                   nbObjects * sizeof(BaseObject),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        primitives = nullptr;
    }

    // =========================
    // Build BVH
    // =========================
    bvhScene = BVHScene::buildBVHScene(&primitivesGPU,
                                       &spheresGPU,
                                       &triangleMeshesGPU,
                                       &implicitSpheresGPU);

    // =========================
    // Upload spheres
    // =========================
    nbSpheres = spheresGPU.size();
    if (nbSpheres > 0)
    {
        cudaMalloc(&spheres, nbSpheres * sizeof(Sphere));
        cudaMemcpy(spheres,
                   spheresGPU.data(),
                   nbSpheres * sizeof(Sphere),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        spheres = nullptr;
    }

    // =========================
    // Upload planes
    // =========================
    nbPlanes = planesGPU.size();
    if (nbPlanes > 0)
    {
        cudaMalloc(&planes, nbPlanes * sizeof(Plane));
        cudaMemcpy(planes,
                   planesGPU.data(),
                   nbPlanes * sizeof(Plane),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        planes = nullptr;
    }

    // =========================
    // Upload meshes
    // =========================
    nbTriangleMeshes = triangleMeshesGPU.size();
    if (nbTriangleMeshes > 0)
    {
        cudaMalloc(&triangleMeshes,
                   nbTriangleMeshes * sizeof(TriangleMesh));

        cudaMemcpy(triangleMeshes,
                   triangleMeshesGPU.data(),
                   nbTriangleMeshes * sizeof(TriangleMesh),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        triangleMeshes = nullptr;
    }
               
    nbImplicitSpheres = implicitSpheresGPU.size();
    if (nbImplicitSpheres > 0)
    {
        cudaMalloc(&implicitSpheres,
                   nbImplicitSpheres * sizeof(ImplicitSphere));
        cudaMemcpy(implicitSpheres,
                   implicitSpheresGPU.data(),
                   nbImplicitSpheres * sizeof(ImplicitSphere),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        implicitSpheres = nullptr;
    }

    bvhScene.d_primitives = primitives;
    bvhScene.d_spheres    = spheres;
    bvhScene.d_planes     = planes;
    bvhScene.d_meshes     = triangleMeshes;
    bvhScene.d_implicitSpheres = implicitSpheres;
}

HOST
void CudaScene::uploadLights(
                             std::vector<Light> lightsGPU){
    nbLights = lightsGPU.size();
    if (nbLights > 0)
    {
        cudaMalloc(&lights,
                   nbLights * sizeof(Light));

        cudaMemcpy(lights,
                   lightsGPU.data(),
                   nbLights * sizeof(Light),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        lights = nullptr;
    }
}

HOST
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

void CudaScene::sceneSize(std::vector<BaseObject> primitivesGPU)
{
    printf("Size of one BVH node: %zu bytes\n", sizeof(BVHSceneNode));
    printf("Size of AABB: %zu bytes\n", sizeof(AABB));
    printf("Size of BaseObject: %zu bytes\n", sizeof(BaseObject));
    printf("Size of Sphere: %zu bytes\n", sizeof(Sphere));
    printf("Size of Plane: %zu bytes\n", sizeof(Plane));
    printf("Size of Material: %zu bytes\n", sizeof(Material));
    printf("Size of Light: %zu bytes\n", sizeof(Light));
    printf("Size of ImplicitSphere: %zu bytes\n", sizeof(ImplicitSphere));
    printf("\n");
    size_t totalSize = 0;
    totalSize += nbSpheres * sizeof(Sphere);
    printf("Size of spheres: %zu bytes. %2.2f gain compared to v1\n", nbSpheres * sizeof(Sphere), (1.f - (nbSpheres * sizeof(Sphere) / 38240.f)) * 100.f);
    totalSize += nbPlanes * sizeof(Plane);
    printf("Size of planes: %zu bytes. %2.2f gain compared to v1\n", nbPlanes * sizeof(Plane), (1.f - (nbPlanes * sizeof(Plane) / 20.f))* 100.f);
    totalSize += nbTriangleMeshes * sizeof(TriangleMesh);
    printf("Size of triangle meshes: %zu bytes. %2.2f gain compared to v1\n", nbTriangleMeshes * sizeof(TriangleMesh), (1.f - (nbTriangleMeshes * sizeof(TriangleMesh) / 1.f)) * 100.f);
    totalSize += nbMaterials * sizeof(Material);
    printf("Size of materials: %zu bytes. %2.2f gain compared to v1\n", nbMaterials * sizeof(Material), (1.f - (nbMaterials * sizeof(Material) / 15360.f)) * 100.f);
    totalSize += nbLights * sizeof(Light);
    printf("Size of lights: %zu bytes. %2.2f gain compared to v1\n", nbLights * sizeof(Light), (1.f - (nbLights * sizeof(Light) / 192.f)) * 100.f);
    totalSize += primitivesGPU.size() * sizeof(BaseObject);
    printf("Size of primitives: %zu bytes. %2.2f gain compared to v1\n", primitivesGPU.size() * sizeof(BaseObject), (1.f - (primitivesGPU.size() * sizeof(BaseObject) / 22992.f)) * 100.f);
    totalSize += bvhScene.getDeviceSize();
    printf("Size of implicit spheres: %zu bytes. %2.2f gain compared to v1\n", nbImplicitSpheres * sizeof(ImplicitSphere), (1.f - (nbImplicitSpheres * sizeof(ImplicitSphere) / 400.f)) * 100.f);
    totalSize += nbImplicitSpheres * sizeof(ImplicitSphere);
    printf("BVH size: %zu bytes. %2.2f gain compared to v1\n", bvhScene.getDeviceSize(), (1.f - (bvhScene.getDeviceSize() / 68928.f)) * 100.f);
    printf("Total size of GPU data: %zu bytes. %2.2f gain compared to v1\n", totalSize, (1.f - (totalSize / 145732.f)) * 100.f);
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

        planesGPU.push_back(p);
    }

    // ===== MATERIALS DE BASE =====
    Material blueGlass = Material::makeMaterial(
        make_float3(0.35f, 0.65f, 1.0f),
        TRANSPARENT,
        0.f,
        0.f,
        1.5f,
        0.f
    );
    Material mirror      = Material::makeMaterial(make_float3(1.f, 1.f, 0.f), MIRROR);
    Material transparent = Material::makeMaterial(make_float3(1.f, 0.f, 1.f), TRANSPARENT, 0.f, 0.f, 1.5f);
    Material emissive    = Material::makeMaterial(make_float3(1.f, 0.f, 0.f), EMISSIVE, 0.f, 0.f, 1.f, 11.f);
    int blueTransparentIdx = materialsGPU.size(); materialsGPU.push_back(blueGlass);
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
            double choose_mat = randomDouble();

            float3 center = make_float3(
                i + 0.9f * randomFloat(),
                0.2f,
                j + 0.9f * randomFloat()
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
            if (choose_mat < 0.40)
            {
                // Lambert coloré
                Material mat = Material::randomLambert();

                materialsGPU.push_back(mat);
                s.materialIndex = materialsGPU.size() - 1;
            }
            else if (choose_mat < 0.62)
            {
                // Métal coloré
                Material mat = Material::randomMetal();

                materialsGPU.push_back(mat);
                s.materialIndex = materialsGPU.size() - 1;
            }
            else if (choose_mat < 0.82)
            {
                // Plastique coloré
                Material mat = Material::randomPlastic();

                materialsGPU.push_back(mat);
                s.materialIndex = materialsGPU.size() - 1;
            }
            else if (choose_mat < 0.90)
            {
                // Miroir légèrement bleuté
                s.materialIndex = mirrorIdx;
            }
            else if (choose_mat < 0.985)
            {
                // Verre coloré aléatoire
                Material mat = Material::randomTransparent();

                materialsGPU.push_back(mat);
                s.materialIndex = materialsGPU.size() - 1;
            }
            else
            {
                // Émissif coloré rare
                Material mat = Material::randomEmissive();

                materialsGPU.push_back(mat);
                s.materialIndex = materialsGPU.size() - 1;

                Light light;
                light.metadata = Light::packMetadata(
                    LightType::SPHERE_GEOM,
                    (int)spheresGPU.size()
                );

                lightsGPU.push_back(light);
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
    gpuScene.uploadObjects(spheresGPU, planesGPU, triangleMeshesGPU, primitivesGPU, std::vector<ImplicitSphere>());
    gpuScene.uploadLights(lightsGPU);
    gpuScene.uploadMaterials(materialsGPU);

    gpuScene.sceneSize(primitivesGPU);
    return gpuScene;
}

CudaScene implicitSpheresScene(float4 sunDir)
{
    CudaScene gpuScene;

    std::vector<ImplicitSphere> implicitSpheresGPU;
    std::vector<Plane> planesGPU;
    std::vector<TriangleMesh> triangleMeshesGPU;
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
            double choose_mat = randomDouble();

            float3 center = make_float3(
                i + 0.9f * randomFloat(),
                0.2f,
                j + 0.9f * randomFloat()
            );

            if (
                length(center - make_float3(4.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(0.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(-4.f, 1.f, 0.f)) < minDist)
                continue;

            ImplicitSphere s;
            s.center1 = center;
            s.center2 = center;
            s.radius = 0.2f;

            // ===== MATERIAL =====
            if (choose_mat < 0.6)
            {
                Material mat = Material::makeMaterial(
                    make_float3(
                        randomFloat(),
                        randomFloat(),
                        randomFloat()
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
                ObjectType::IMPLICIT_SPHERE,
                (int)implicitSpheresGPU.size()
            });
            implicitSpheresGPU.push_back(s);
        }
    }

    // ===== GROSSES SPHERES =====
    auto addBigSphere = [&](float3 center, float radius, int matIndex)
    {
        ImplicitSphere s;
        s.center1 = center;
        s.center2 = center;
        s.radius = radius;
        s.materialIndex = matIndex;

        float3 r = make_float3(radius);

        primitivesGPU.push_back(BaseObject{
            center - r,
            center + r,
            ObjectType::IMPLICIT_SPHERE,
            (int)implicitSpheresGPU.size()
        });

        implicitSpheresGPU.push_back(s);
        Material& mat = materialsGPU[matIndex];

        if (mat.type() == MaterialType::EMISSIVE)
        {
            Light l;
            l.metadata = Light::packMetadata(LightType::IMPLICIT_SPHERE_GEOM, implicitSpheresGPU.size() - 1);
            lightsGPU.push_back(l);
        }
    };
    /*
    MeshAndPrimitive meshAndPrim = loadTriangleMesh("../data/bunny/Bunny.obj", materialsGPU.size() - 1, triangleMeshesGPU.size());
    triangleMeshesGPU.push_back(meshAndPrim.mesh);
    primitivesGPU.push_back(meshAndPrim.prim);
    */

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
    gpuScene.uploadObjects(std::vector<Sphere>(), planesGPU, triangleMeshesGPU, primitivesGPU, implicitSpheresGPU);
    gpuScene.uploadLights(lightsGPU);
    gpuScene.uploadMaterials(materialsGPU);

    gpuScene.sceneSize(primitivesGPU);
    return gpuScene;
}

CudaScene singleObject(float4 sunDir)
{
    CudaScene gpuScene;
    std::vector<TriangleMesh> triangleMeshesGPU;
    std::vector<BaseObject> primitivesGPU;
    std::vector<Material> materialsGPU;
    std::vector<Light> lightsGPU;
    Material mat = Material::makeMaterial(
                    make_float3(
                        randomFloat(),
                        randomFloat(),
                        randomFloat()
                    ),
                    LAMBERT,
                    1.0f
                );
    materialsGPU.push_back(mat);
    Quaternion rotation = quaternionFromAxisAngle(make_float3(0.f, 1.f, 0.f), 00.f);
    MeshAndPrimitive meshAndPrim = loadTriangleMesh("../data/bunny/Bunny.obj", materialsGPU.size() - 1, triangleMeshesGPU.size(), make_float3(2.f, 2.f, 2.f), rotation, make_float3(0.f, 0.f, 0.f));
    triangleMeshesGPU.push_back(meshAndPrim.mesh);
    primitivesGPU.push_back(meshAndPrim.prim);
    Light l;
    l.color_power = make_float4(1.f, 0.95f, 0.9f, 100.f);
    l.direction = make_float4(sunDir.x, sunDir.y, sunDir.z, 0.f);
    l.metadata = Light::packMetadata(LightType::SUN, 0);
    lightsGPU.push_back(l);
    gpuScene.uploadObjects(std::vector<Sphere>(), std::vector<Plane>(), triangleMeshesGPU, primitivesGPU, std::vector<ImplicitSphere>());
    gpuScene.uploadLights(lightsGPU);
    gpuScene.uploadMaterials(materialsGPU);
    gpuScene.sceneSize(primitivesGPU);
    return gpuScene;
}



