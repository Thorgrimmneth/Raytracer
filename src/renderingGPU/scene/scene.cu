#include "../camera/camera.cuh"
#include "scene.cuh"
#include "scene_helper.cuh"

HOST void CudaScene::uploadObjects(CudaSceneHelper &helper)
{
    // =========================
    // Upload primitives
    // =========================
    int nbObjects = helper.primitivesGPU.size();

    if (nbObjects > 0)
    {
        cudaMalloc(&primitives, nbObjects * sizeof(BaseObject));
        cudaMemcpy(primitives, helper.primitivesGPU.data(), nbObjects * sizeof(BaseObject), cudaMemcpyHostToDevice);
    }
    else
    {
        primitives = nullptr;
    }

    // =========================
    // Build BVH
    // =========================
    bvhScene = BVHScene::buildBVHScene(&helper.primitivesGPU, &helper.spheresGPU, &helper.triangleMeshesGPU,
                                       &helper.implicitSpheresGPU);

    // =========================
    // Upload spheres
    // =========================
    nbSpheres = helper.spheresGPU.size();
    if (nbSpheres > 0)
    {
        cudaMalloc(&spheres, nbSpheres * sizeof(Sphere));
        cudaMemcpy(spheres, helper.spheresGPU.data(), nbSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);
    }
    else
    {
        spheres = nullptr;
    }

    // =========================
    // Upload planes
    // =========================
    nbPlanes = helper.planesGPU.size();
    if (nbPlanes > 0)
    {
        cudaMalloc(&planes, nbPlanes * sizeof(Plane));
        cudaMemcpy(planes, helper.planesGPU.data(), nbPlanes * sizeof(Plane), cudaMemcpyHostToDevice);
    }
    else
    {
        planes = nullptr;
    }

    // =========================
    // Upload meshes
    // =========================
    nbTriangleMeshes = helper.triangleMeshesGPU.size();
    if (nbTriangleMeshes > 0)
    {
        cudaMalloc(&triangleMeshes, nbTriangleMeshes * sizeof(TriangleMesh));

        cudaMemcpy(triangleMeshes, helper.triangleMeshesGPU.data(), nbTriangleMeshes * sizeof(TriangleMesh),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        triangleMeshes = nullptr;
    }

    nbImplicitSpheres = helper.implicitSpheresGPU.size();
    if (nbImplicitSpheres > 0)
    {
        cudaMalloc(&implicitSpheres, nbImplicitSpheres * sizeof(ImplicitSphere));
        cudaMemcpy(implicitSpheres, helper.implicitSpheresGPU.data(), nbImplicitSpheres * sizeof(ImplicitSphere),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        implicitSpheres = nullptr;
    }

    bvhScene.d_primitives = primitives;
    bvhScene.d_spheres = spheres;
    bvhScene.d_planes = planes;
    bvhScene.d_meshes = triangleMeshes;
    bvhScene.d_implicitSpheres = implicitSpheres;
}

HOST void CudaScene::uploadLights(CudaSceneHelper &helper)
{
    nbLights = helper.lightsGPU.size();

    if (nbLights == 0)
    {
        lights = nullptr;
        lightProbabilities = nullptr;
        lightCumulativeWeights = nullptr;
        return;
    }

    std::vector<float> weights(nbLights);
    std::vector<float> probabilities(nbLights);
    std::vector<float> cumulativeWeights(nbLights);

    float totalWeight = 0.0f;

    // Compute weights
    for (int i = 0; i < nbLights; ++i)
    {
        const Light &light = helper.lightsGPU[i];

        float weight = length(light.getColorPower());

        // Avoid zero-weight lights
        weight = fmaxf(weight, 1e-8f);

        weights[i] = weight;
        totalWeight += weight;
    }

    // Build cumulative weights and normalized probabilities
    float cumulative = 0.0f;

    for (int i = 0; i < nbLights; ++i)
    {
        cumulative += weights[i];

        cumulativeWeights[i] = cumulative;
        probabilities[i] = weights[i] / totalWeight;
    }

    cudaMalloc(&lights, nbLights * sizeof(Light));
    cudaMemcpy(lights, helper.lightsGPU.data(), nbLights * sizeof(Light), cudaMemcpyHostToDevice);

    cudaMalloc(&lightCumulativeWeights, nbLights * sizeof(float));
    cudaMemcpy(lightCumulativeWeights, cumulativeWeights.data(), nbLights * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&lightProbabilities, nbLights * sizeof(float));
    cudaMemcpy(lightProbabilities, probabilities.data(), nbLights * sizeof(float), cudaMemcpyHostToDevice);
}

HOST void CudaScene::uploadMaterials(CudaSceneHelper &helper)
{
    nbMaterials = helper.materialsGPU.size();

    if (nbMaterials > 0)
    {
        cudaMalloc(&materials, nbMaterials * sizeof(Material));

        cudaMemcpy(materials, helper.materialsGPU.data(), nbMaterials * sizeof(Material), cudaMemcpyHostToDevice);
    }
    else
    {
        materials = nullptr;
    }
}

void CudaScene::sceneSize(CudaSceneHelper &helper)
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
    printf("Size of spheres: %zu bytes. %2.2f gain compared to v1\n", nbSpheres * sizeof(Sphere),
           (1.f - (nbSpheres * sizeof(Sphere) / 38240.f)) * 100.f);
    totalSize += nbPlanes * sizeof(Plane);
    printf("Size of planes: %zu bytes. %2.2f gain compared to v1\n", nbPlanes * sizeof(Plane),
           (1.f - (nbPlanes * sizeof(Plane) / 20.f)) * 100.f);
    totalSize += nbTriangleMeshes * sizeof(TriangleMesh);
    printf("Size of triangle meshes: %zu bytes. %2.2f gain compared to v1\n", nbTriangleMeshes * sizeof(TriangleMesh),
           (1.f - (nbTriangleMeshes * sizeof(TriangleMesh) / 1.f)) * 100.f);
    totalSize += nbMaterials * sizeof(Material);
    printf("Size of materials: %zu bytes. %2.2f gain compared to v1\n", nbMaterials * sizeof(Material),
           (1.f - (nbMaterials * sizeof(Material) / 15360.f)) * 100.f);
    totalSize += nbLights * sizeof(Light);
    printf("Size of lights: %zu bytes. %2.2f gain compared to v1\n", nbLights * sizeof(Light),
           (1.f - (nbLights * sizeof(Light) / 192.f)) * 100.f);
    totalSize += helper.primitivesGPU.size() * sizeof(BaseObject);
    printf("Size of primitives: %zu bytes. %2.2f gain compared to v1\n",
           helper.primitivesGPU.size() * sizeof(BaseObject),
           (1.f - (helper.primitivesGPU.size() * sizeof(BaseObject) / 22992.f)) * 100.f);
    totalSize += bvhScene.getDeviceSize();
    printf("Size of implicit spheres: %zu bytes. %2.2f gain compared to v1\n",
           nbImplicitSpheres * sizeof(ImplicitSphere),
           (1.f - (nbImplicitSpheres * sizeof(ImplicitSphere) / 400.f)) * 100.f);
    totalSize += nbImplicitSpheres * sizeof(ImplicitSphere);
    printf("BVH size: %zu bytes. %2.2f gain compared to v1\n", bvhScene.getDeviceSize(),
           (1.f - (bvhScene.getDeviceSize() / 68928.f)) * 100.f);
    printf("Total size of GPU data: %zu bytes. %2.2f gain compared to v1\n", totalSize,
           (1.f - (totalSize / 145732.f)) * 100.f);
}

void sortMaterials(CudaSceneHelper &helper)
{
    int padding[6];
    padding[0] = 0; // Account for ground plane at index 0
    padding[1] = helper.lambertList.size();
    padding[2] = padding[1] + helper.metalList.size();
    padding[3] = padding[2] + helper.plasticList.size();
    padding[4] = padding[3] + helper.transparentList.size();
    padding[5] = padding[4] + helper.emissiveList.size();

    // adding materials from a same type together to improve memory coherence when shading
    helper.materialsGPU.insert(helper.materialsGPU.end(), helper.lambertList.begin(), helper.lambertList.end());
    helper.materialsGPU.insert(helper.materialsGPU.end(), helper.metalList.begin(), helper.metalList.end());
    helper.materialsGPU.insert(helper.materialsGPU.end(), helper.plasticList.begin(), helper.plasticList.end());
    helper.materialsGPU.insert(helper.materialsGPU.end(), helper.transparentList.begin(), helper.transparentList.end());
    helper.materialsGPU.insert(helper.materialsGPU.end(), helper.emissiveList.begin(), helper.emissiveList.end());
    helper.materialsGPU.insert(helper.materialsGPU.end(), helper.mirrorList.begin(), helper.mirrorList.end());

    for (int i = 0; i < helper.planesGPU.size(); i++)
    {
        helper.planesGPU[i].materialIndex = helper.planesGPU[i].materialIndex + padding[helper.planeType[i]];
    }

    for (int i = 0; i < helper.spheresGPU.size(); i++)
    {
        helper.spheresGPU[i].setMaterialIndex(helper.spheresGPU[i].getMaterialIndex() + padding[helper.sphereType[i]]);
    }

    for (int i = 0; i < helper.triangleMeshesGPU.size(); i++)
    {
        helper.triangleMeshesGPU[i].materialIndex =
            helper.triangleMeshesGPU[i].materialIndex + padding[helper.triangleMeshType[i]];
    }

    for (int i = 0; i < helper.implicitSpheresGPU.size(); i++)
    {
        helper.implicitSpheresGPU[i].setMaterialIndex(helper.implicitSpheresGPU[i].getMaterialIndex() +
                                                      padding[helper.sphereType[i]]);
    }
}

CudaScene spheresScene(float4 sunDir)
{
    CudaScene gpuScene;
    CudaSceneHelper helper;

    // ===== PLAN =====
    {
        Plane p = Plane(make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f));

        Material ground = Material::makeMaterial(make_float3(0.5f), LAMBERT, 1.0f);
        helper.lambertList.push_back(ground);
        p.materialIndex = helper.lambertList.size() - 1;
        helper.planeType.push_back(0);
        helper.planesGPU.push_back(p);
    }

    // ===== MATERIALS DE BASE =====
    Material blueGlass = Material::makeMaterial(make_float3(0.35f, 0.65f, 1.0f), TRANSPARENT, 0.f, 0.f, 1.5f, 0.f);

    Material mirror = Material::makeMaterial(make_float3(1.f, 1.f, 1.f), MIRROR);
    Material transparent = Material::makeMaterial(make_float3(0.9f, 0.9f, 0.9f), TRANSPARENT, 0.f, 0.f, 1.5f);
    Material emissive = Material::makeMaterial(make_float3(1.f, 0.f, 0.f), EMISSIVE, 0.f, 0.f, 1.f, 11.f);
    int blueTransparentIdx = helper.transparentList.size();
    helper.transparentList.push_back(blueGlass);
    int mirrorIdx = helper.mirrorList.size();
    helper.mirrorList.push_back(mirror);
    int transparentIdx = helper.transparentList.size();
    helper.transparentList.push_back(transparent);
    int emissiveIdx = helper.emissiveList.size();
    helper.emissiveList.push_back(emissive);

    float bigRadius = 1.0f;
    float smallRadius = 0.2f;
    float margin = 0.05f;
    float minDist = bigRadius + smallRadius + margin;

    int numberOfSpheresPerSide = 10;
    // ===== PETITES SPHERES =====
    for (int i = -numberOfSpheresPerSide; i < numberOfSpheresPerSide; i++)
    {
        for (int j = -numberOfSpheresPerSide; j < numberOfSpheresPerSide; j++)
        {
            double choose_mat = randomDouble();

            float3 center = make_float3(i + 0.9f * randomFloat(), 0.2f, j + 0.9f * randomFloat());

            if (length(center - make_float3(4.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(0.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(-4.f, 1.f, 0.f)) < minDist)
                continue;

            Sphere s = Sphere(center, smallRadius);

            // ===== MATERIAL =====
            if (choose_mat < 0.40)
            {
                // Lambert coloré
                Material mat = Material::randomLambert();

                helper.lambertList.push_back(mat);
                s.setMaterialIndex(helper.lambertList.size() - 1);
                helper.sphereType.push_back(0);
            }
            else if (choose_mat < 0.62)
            {
                // Métal coloré
                Material mat = Material::randomMetal();

                helper.metalList.push_back(mat);
                s.setMaterialIndex(helper.metalList.size() - 1);
                helper.sphereType.push_back(1);
            }
            else if (choose_mat < 0.82)
            {
                // Plastique coloré
                Material mat = Material::randomPlastic();

                helper.plasticList.push_back(mat);
                s.setMaterialIndex(helper.plasticList.size() - 1);
                helper.sphereType.push_back(2);
            }
            else if (choose_mat < 0.90)
            {
                // Miroir légèrement bleuté
                s.setMaterialIndex(mirrorIdx);
                helper.sphereType.push_back(5);
            }
            else if (choose_mat < 0.985)
            {
                // Verre coloré aléatoire
                Material mat = Material::randomTransparent();

                helper.transparentList.push_back(mat);
                s.setMaterialIndex(helper.transparentList.size() - 1);
                helper.sphereType.push_back(3);
            }
            else
            {
                // Émissif coloré rare
                Material mat = Material::randomEmissive();

                helper.emissiveList.push_back(mat);
                s.setMaterialIndex(helper.emissiveList.size() - 1);
                helper.sphereType.push_back(4);
                Light light;
                light.metadata = Light::packMetadata(LightType::SPHERE_GEOM, (int)helper.spheresGPU.size());

                helper.lightsGPU.push_back(light);
            }

            // ===== AABB =====
            float3 r = make_float3(s.getRadius());

            helper.primitivesGPU.push_back(
                BaseObject{center - r, center + r, ObjectType::SPHERE, (int)helper.spheresGPU.size()});
            helper.spheresGPU.push_back(s);
        }
    }

    // ===== GROSSES SPHERES =====
    auto addBigSphere = [&](float3 center, float radius, int matIndex, int sphereTypeValue) {
        Sphere s = Sphere(center, radius, matIndex);

        float3 r = make_float3(radius);

        helper.primitivesGPU.push_back(
            BaseObject{center - r, center + r, ObjectType::SPHERE, (int)helper.spheresGPU.size()});

        helper.spheresGPU.push_back(s);
        helper.sphereType.push_back(sphereTypeValue);

        if (sphereTypeValue == 4) // EMISSIVE
        {
            Light l;
            l.metadata = Light::packMetadata(LightType::SPHERE_GEOM, helper.spheresGPU.size() - 1);
            helper.lightsGPU.push_back(l);
        }
    };

    addBigSphere(make_float3(0.f, 1.f, 0.f), 1.f, transparentIdx, 3);
    addBigSphere(make_float3(-4.f, 1.f, 0.f), 1.f, emissiveIdx, 4);
    addBigSphere(make_float3(4.f, 1.f, 0.f), 1.f, mirrorIdx, 5);

    sortMaterials(helper);

    // ===== LIGHT (SUN) =====
    Light l;
    l.color_power = make_float4(1.f, 0.95f, 0.9f, 100.f);
    l.direction = make_float4(sunDir.x, sunDir.y, sunDir.z, 0.f);
    l.metadata = Light::packMetadata(LightType::SUN, 0);
    helper.lightsGPU.push_back(l);

    // ===== UPLOAD =====
    gpuScene.uploadObjects(helper);
    gpuScene.uploadLights(helper);
    gpuScene.uploadMaterials(helper);

    gpuScene.sceneSize(helper);
    return gpuScene;
}

CudaScene implicitSpheresScene(float4 sunDir)
{
    CudaScene gpuScene;
    CudaSceneHelper helper;

    // ===== PLAN =====
    {
        Plane p = Plane(make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f));

        Material ground = Material::makeMaterial(make_float3(0.5f), LAMBERT, 1.0f);
        helper.lambertList.push_back(ground);
        p.materialIndex = helper.lambertList.size() - 1;
        helper.planeType.push_back(0);
        helper.planesGPU.push_back(p);
    }

    // ===== MATERIALS DE BASE =====
    Material blueGlass = Material::makeMaterial(make_float3(0.35f, 0.65f, 1.0f), TRANSPARENT, 0.f, 0.f, 1.5f, 0.f);

    Material mirror = Material::makeMaterial(make_float3(1.f, 1.f, 1.f), MIRROR);
    Material transparent = Material::makeMaterial(make_float3(0.9f, 0.9f, 0.9f), TRANSPARENT, 0.f, 0.f, 1.5f);
    Material emissive = Material::makeMaterial(make_float3(1.f, 0.f, 0.f), EMISSIVE, 0.f, 0.f, 1.f, 11.f);
    int blueTransparentIdx = helper.transparentList.size();
    helper.transparentList.push_back(blueGlass);
    int mirrorIdx = helper.mirrorList.size();
    helper.mirrorList.push_back(mirror);
    int transparentIdx = helper.transparentList.size();
    helper.transparentList.push_back(transparent);
    int emissiveIdx = helper.emissiveList.size();
    helper.emissiveList.push_back(emissive);

    float bigRadius = 1.0f;
    float smallRadius = 0.2f;
    float margin = 0.05f;
    float minDist = bigRadius + smallRadius + margin;

    int numberOfSpheresPerSide = 10;
    // ===== PETITES SPHERES =====
    for (int i = -numberOfSpheresPerSide; i < numberOfSpheresPerSide; i++)
    {
        for (int j = -numberOfSpheresPerSide; j < numberOfSpheresPerSide; j++)
        {
            double choose_mat = randomDouble();

            float3 center = make_float3(i + 0.9f * randomFloat(), 0.2f, j + 0.9f * randomFloat());

            if (length(center - make_float3(4.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(0.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(-4.f, 1.f, 0.f)) < minDist)
                continue;

            ImplicitSphere s = ImplicitSphere(center, smallRadius, center, 0);

            // ===== MATERIAL =====
            if (choose_mat < 0.40)
            {
                // Lambert coloré
                Material mat = Material::randomLambert();

                helper.lambertList.push_back(mat);
                s.setMaterialIndex(helper.lambertList.size() - 1);
                helper.sphereType.push_back(0);
            }
            else if (choose_mat < 0.62)
            {
                // Métal coloré
                Material mat = Material::randomMetal();

                helper.metalList.push_back(mat);
                s.setMaterialIndex(helper.metalList.size() - 1);
                helper.sphereType.push_back(1);
            }
            else if (choose_mat < 0.82)
            {
                // Plastique coloré
                Material mat = Material::randomPlastic();

                helper.plasticList.push_back(mat);
                s.setMaterialIndex(helper.plasticList.size() - 1);
                helper.sphereType.push_back(2);
            }
            else if (choose_mat < 0.90)
            {
                // Miroir légèrement bleuté
                s.setMaterialIndex(mirrorIdx);
                helper.sphereType.push_back(5);
            }
            else if (choose_mat < 0.985)
            {
                // Verre coloré aléatoire
                Material mat = Material::randomTransparent();

                helper.transparentList.push_back(mat);
                s.setMaterialIndex(helper.transparentList.size() - 1);
                helper.sphereType.push_back(3);
            }
            else
            {
                // Émissif coloré rare
                Material mat = Material::randomEmissive();

                helper.emissiveList.push_back(mat);
                s.setMaterialIndex(helper.emissiveList.size() - 1);
                helper.sphereType.push_back(4);
                Light light;
                light.metadata =
                    Light::packMetadata(LightType::IMPLICIT_SPHERE_GEOM, (int)helper.implicitSpheresGPU.size());

                helper.lightsGPU.push_back(light);
            }

            // ===== AABB =====
            float3 r = make_float3(s.getRadius());

            helper.primitivesGPU.push_back(
                BaseObject{center - r, center + r, ObjectType::IMPLICIT_SPHERE, (int)helper.implicitSpheresGPU.size()});
            helper.implicitSpheresGPU.push_back(s);
        }
    }

    // ===== GROSSES SPHERES =====
    auto addBigSphere = [&](float3 center, float radius, int matIndex, int sphereTypeValue) {
        ImplicitSphere s = ImplicitSphere(center, radius, center, matIndex);

        float3 r = make_float3(s.getRadius());

        helper.primitivesGPU.push_back(
            BaseObject{center - r, center + r, ObjectType::IMPLICIT_SPHERE, (int)helper.implicitSpheresGPU.size()});

        helper.implicitSpheresGPU.push_back(s);
        helper.sphereType.push_back(sphereTypeValue);

        if (sphereTypeValue == 4) // EMISSIVE
        {
            Light l;
            l.metadata = Light::packMetadata(LightType::IMPLICIT_SPHERE_GEOM, helper.implicitSpheresGPU.size() - 1);
            helper.lightsGPU.push_back(l);
        }
    };

    addBigSphere(make_float3(0.f, 1.f, 0.f), 1.f, transparentIdx, 3);
    addBigSphere(make_float3(-4.f, 1.f, 0.f), 1.f, emissiveIdx, 4);
    addBigSphere(make_float3(4.f, 1.f, 0.f), 1.f, mirrorIdx, 5);

    sortMaterials(helper);

    // ===== LIGHT (SUN) =====
    Light l;
    l.color_power = make_float4(1.f, 0.95f, 0.9f, 100.f);
    l.direction = make_float4(sunDir.x, sunDir.y, sunDir.z, 0.f);
    l.metadata = Light::packMetadata(LightType::SUN, 0);
    helper.lightsGPU.push_back(l);

    // ===== UPLOAD =====
    gpuScene.uploadObjects(helper);
    gpuScene.uploadLights(helper);
    gpuScene.uploadMaterials(helper);

    gpuScene.sceneSize(helper);
    return gpuScene;
}

CudaScene singleObject(float4 sunDir)
{
    CudaScene gpuScene;
    CudaSceneHelper helper;
    Material mirror = Material::makeMaterial(make_float3(1.f, 1.f, 1.f), MIRROR);
    Material transparent = Material::makeMaterial(make_float3(0.9f, 0.9f, 0.9f), TRANSPARENT, 0.f, 0.f, 1.5f);
    helper.materialsGPU.push_back(mirror);
    helper.materialsGPU.push_back(transparent);
    Material mat = Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()), LAMBERT, 1.0f);
    helper.materialsGPU.push_back(mat);
    Quaternion rotation = quaternionFromAxisAngle(make_float3(0.f, 1.f, 0.f), 0.f);
    MeshAndPrimitive meshAndPrim =
        loadTriangleMesh("data/bunny/Bunny.obj", 2, helper.triangleMeshesGPU.size(),
                         make_float3(2.f, 2.f, 2.f), rotation, make_float3(0.f, 0.f, 0.f));
    helper.triangleMeshesGPU.push_back(meshAndPrim.mesh);
    helper.primitivesGPU.push_back(meshAndPrim.prim);
    Light l;
    l.color_power = make_float4(1.f, 0.95f, 0.9f, 100.f);
    l.direction = make_float4(sunDir.x, sunDir.y, sunDir.z, 0.f);
    l.metadata = Light::packMetadata(LightType::SUN, 0);
    helper.lightsGPU.push_back(l);

    OptixContext context;
    context.initialize();

    OptixModuleManager raygenModuleManager;

    raygenModuleManager.createFromPath(context.deviceContext, "build/raygen.ptx");

    std::cout << "Optix module created successfully" << std::endl;

    OptixModuleManager missModuleManager;
    missModuleManager.createFromPath(context.deviceContext, "build/miss.ptx");

    OptixModuleManager chitModuleManager;
    chitModuleManager.createFromPath(context.deviceContext, "build/closesthit.ptx");

    OptixProgramGroupManager programGroupManager;
    programGroupManager.create(context.deviceContext, raygenModuleManager.module, missModuleManager.module,
                               chitModuleManager.module);

    std::cout << "RaygenPG = " << programGroupManager.raygenPG << "\nMissPG   = " << programGroupManager.missPG
              << "\nHitPG    = " << programGroupManager.hitPG << std::endl;

    OptixPipelineManager pipelineManager;

    pipelineManager.create(context.deviceContext, raygenModuleManager.getPipelineCompileOptions(),
                           programGroupManager.raygenPG, programGroupManager.missPG, programGroupManager.hitPG);

    std::cout << "Pipeline = " << pipelineManager.pipeline << std::endl;

    OptixSBTManager sbtManager;
    sbtManager.create(programGroupManager.raygenPG, programGroupManager.missPG, programGroupManager.hitPG,
                      meshAndPrim.mesh.vertices, meshAndPrim.mesh.normals, meshAndPrim.mesh.uvs, meshAndPrim.mesh.triangles, meshAndPrim.mesh.materialIndex);

    std::cout << "raygenRecord = " << sbtManager.sbt.raygenRecord << "\nmissCount = " << sbtManager.sbt.missRecordCount
              << "\nhitCount = " << sbtManager.sbt.hitgroupRecordCount << std::endl;

    float3 camPos = make_float3(8.f, 2.f, 3.f);
    float3 camTarget = make_float3(0.f, 0.f, 0.f);
    float3 camUp = make_float3(0.f, 1.f, 0.f);

    float fov = 60.f;
    float aspect = (float)1920 / (float)1080;
    float focalDistance = 1.f;

    // === Base vectors EXACTEMENT comme CPU ===
    float3 w = normalize(camPos - camTarget);
    float3 u = normalize(cross(camUp, w));
    float3 v = normalize(cross(w, u));

    // === Viewport ===
    float theta = fov * 3.14159265f / 180.f;
    float viewportHeight = 2.f * tanf(theta * 0.5f) * focalDistance;
    float viewportWidth = viewportHeight * aspect;

    float3 viewportU = u * viewportWidth;
    float3 viewportV = v * viewportHeight;

    float3 topLeft = camPos - w * focalDistance + viewportV * 0.5f - viewportU * 0.5f;
    Camera camera = Camera{make_float4(camPos, 1.f), make_float4(topLeft, 1.f), make_float4(viewportU, 1.f), make_float4(viewportV, 1.f)};
    OptixLaunchParamsManager launchParamsManager;
    launchParamsManager.create();
    std::cout << "d_params = " << launchParamsManager.d_params << std::endl;

    OPTIX_CHECK(optixPipelineSetStackSize(pipelineManager.pipeline,
                                          2 * 1024, // directCallableStackSizeFromTraversal
                                          2 * 1024, // directCallableStackSizeFromState
                                          2 * 1024, // continuationStackSize
                                          1         // maxTraversableGraphDepth
                                          ));

    OptixGAS gas;
    gas.build(context.deviceContext, context.stream, meshAndPrim.mesh.vertices, meshAndPrim.mesh.vertexCount,
              meshAndPrim.mesh.triangles, meshAndPrim.mesh.triangleCount);
    std::cout << "GAS handle = " << gas.handle << std::endl;
    std::cout << "Vertices  : " << meshAndPrim.mesh.vertexCount << std::endl;

    std::cout << "Triangles : " << meshAndPrim.mesh.triangleCount << std::endl;
    launchParamsManager.params.traversable = gas.handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManager.d_params), &launchParamsManager.params,
                          sizeof(LaunchParams), cudaMemcpyHostToDevice));
    /*
    OPTIX_CHECK(optixLaunch(pipelineManager.pipeline,
                            0, // stream
                            launchParamsManager.d_params, sizeof(LaunchParams), &sbtManager.sbt,
                            launchParamsManager.params.width, launchParamsManager.params.height, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto framebuffer = launchParamsManager.downloadFramebuffer();*/
    gpuScene.optixData = OptixSceneData{pipelineManager.pipeline,   sbtManager.sbt, launchParamsManager.d_params,
                                        launchParamsManager.params, gas.handle,     gas.d_gasBuffer};
    raygenModuleManager.destroy();
    missModuleManager.destroy();
    chitModuleManager.destroy();

    programGroupManager.destroy();

    gpuScene.uploadObjects(helper);
    gpuScene.uploadLights(helper);
    gpuScene.uploadMaterials(helper);
    // gpuScene.sceneSize(helper);
    return gpuScene;
}
