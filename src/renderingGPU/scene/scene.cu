#include "scene.cuh"

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
    // Upload mesh geometries (shared geometry data)
    // =========================
    nbMeshGeometries = helper.meshGeometriesGPU.size();
    if (nbMeshGeometries > 0)
    {
        cudaMalloc(&meshGeometries, nbMeshGeometries * sizeof(MeshGeometry));
        cudaMemcpy(meshGeometries, helper.meshGeometriesGPU.data(), nbMeshGeometries * sizeof(MeshGeometry),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        meshGeometries = nullptr;
    }

    // =========================
    // Upload mesh instances (per-instance data with transforms and materials)
    // =========================
    nbMeshInstances = helper.meshInstancesGPU.size();
    if (nbMeshInstances > 0)
    {
        cudaMalloc(&meshInstances, nbMeshInstances * sizeof(MeshInstance));
        cudaMemcpy(meshInstances, helper.meshInstancesGPU.data(), nbMeshInstances * sizeof(MeshInstance),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        meshInstances = nullptr;
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
        helper.spheresGPU[i].materialIndex = helper.spheresGPU[i].materialIndex + padding[helper.sphereType[i]];
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
                s.materialIndex = helper.lambertList.size() - 1;
                helper.sphereType.push_back(0);
            }
            else if (choose_mat < 0.62)
            {
                // Métal coloré
                Material mat = Material::randomMetal();

                helper.metalList.push_back(mat);
                s.materialIndex = helper.metalList.size() - 1;
                helper.sphereType.push_back(1);
            }
            else if (choose_mat < 0.82)
            {
                // Plastique coloré
                Material mat = Material::randomPlastic();

                helper.plasticList.push_back(mat);
                s.materialIndex = helper.plasticList.size() - 1;
                helper.sphereType.push_back(2);
            }
            else if (choose_mat < 0.90)
            {
                // Miroir légèrement bleuté
                s.materialIndex = mirrorIdx;
                helper.sphereType.push_back(5);
            }
            else if (choose_mat < 0.985)
            {
                // Verre coloré aléatoire
                Material mat = Material::randomTransparent();

                helper.transparentList.push_back(mat);
                s.materialIndex = helper.transparentList.size() - 1;
                helper.sphereType.push_back(3);
            }
            else
            {
                // Émissif coloré rare
                Material mat = Material::randomEmissive();

                helper.emissiveList.push_back(mat);
                s.materialIndex = helper.emissiveList.size() - 1;
                helper.sphereType.push_back(4);
                Light light;
                light.metadata = Light::packMetadata(LightType::SPHERE_GEOM, (int)helper.spheresGPU.size());

                helper.lightsGPU.push_back(light);
            }

            // ===== AABB =====
            float3 r = make_float3(s.radius);

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

    return gpuScene;
}

CudaScene singleObject(float4 sunDir, int rngmanip)
{
    CudaScene gpuScene;
    CudaSceneHelper helper;
    Light sun = createSun(sunDir, helper);
    helper.lightsGPU.push_back(sun);

    for (int i = 0; i < rngmanip; i++)
    {
        float manipRNG = randomFloat();
    }
    
    for (int i = 0; i < 5; i++)
    {
        Material emissive = Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()), EMISSIVE,
                                                   0.f, 0.f, 1.f, randomFloat() * 5.f + 8.f);
        helper.materialsGPU.push_back(emissive);
    }
    for (int i = 0; i < 10; i++)
    {
        Material mirror = Material::makeMaterial(make_float3(1.f), MIRROR);
        helper.materialsGPU.push_back(mirror);
    }

    for (int i = 0; i < 10; i++)
    {
        Material transparent = Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()),
                                                      TRANSPARENT, 0.f, 0.f, 1.5f);
        helper.materialsGPU.push_back(transparent);
    }
    for (int i = 0; i < 15; i++)
    {
        Material lambert =
            Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()), LAMBERT, 1.0f);
        helper.materialsGPU.push_back(lambert);
    }

    for (int i = 0; i < 10; i++)
    {
        float roughness = randomFloat() * 0.5f;
        Material metal = Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()), METAL,
                                                roughness * roughness, 1.f);
        helper.materialsGPU.push_back(metal);
    }
    
    for (int i = 0; i < 10; i++)
    {
        float roughness = randomFloat() * 0.5f;
        Material plastic = Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()), PLASTIC,
                                                  roughness * roughness);
        helper.materialsGPU.push_back(plastic);
    }

    // ===== MESH INSTANCING: Load geometry once, create multiple instances =====
    MeshGeometry bunnyGeometry = loadMeshGeometry("data/bunny/Bunny.obj");
    helper.meshGeometriesGPU.push_back(bunnyGeometry);
    MeshGeometry dragonGeometry = loadMeshGeometry("data/dragon/dragon.obj", make_float3(10.f));
    helper.meshGeometriesGPU.push_back(dragonGeometry);

    // Create 50 instances with different transforms and materials
    for (int i = 0; i < 50; i++)
    {
        Quaternion rotation = quaternionFromAxisAngle(
            make_float3(randomFloat() * 2.f, randomFloat() * 2.f, randomFloat() * 2.f), randomFloat() * 360.f);

        float3 scale = make_float3(randomFloat() * 0.5f + 0.5f);
        float3 translation =
            make_float3(randomFloat() * 10.f - 4.f, randomFloat() * 10.f - 6.f, -randomFloat() * 10.f + 4.f);

        int materialIndex = int(randomFloat() * helper.materialsGPU.size());

        // Create an instance (no GPU allocation here, just structure setup)
        MeshInstance instance = createMeshInstance(int(randomFloat() * helper.meshGeometriesGPU.size()), materialIndex,
                                                   scale, rotation, translation);
        helper.meshInstancesGPU.push_back(instance);
    }

    // Add ground plane
    Plane p = Plane(make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f));
    Material ground = Material::makeMaterial(make_float3(0.5f), LAMBERT, 1.0f);
    helper.materialsGPU.push_back(ground);
    p.materialIndex = helper.materialsGPU.size() - 1;
    helper.meshGeometriesGPU.push_back(PlaneToMesh(p, 20000.f));
    helper.meshInstancesGPU.push_back(createMeshInstance(helper.meshGeometriesGPU.size() - 1, p.materialIndex));

    OptixContext context;
    context.initialize();
    OptixProgramGroupManager programGroupManagerRadiance;
    OptixPipelineManager pipelineManagerRadiance;
    OptixLaunchParamsManager<LaunchRadianceParams> launchParamsManagerRadiance;

    initOptix(context, programGroupManagerRadiance, pipelineManagerRadiance, launchParamsManagerRadiance,
              "build/radianceRaygen.ptx", "__raygen__radiance", "build/radianceMiss.ptx", "__miss__radiance",
              "build/radianceClosestHit.ptx", "__closesthit__radiance");

    OptixSBTManager sbtManagerRadiance;
    sbtManagerRadiance.create(helper.meshGeometriesGPU, programGroupManagerRadiance);

    OptixProgramGroupManager programGroupManagerShadow;
    OptixPipelineManager pipelineManagerShadow;
    OptixLaunchParamsManager<LaunchShadowParams> launchParamsManagerShadow;
    initOptix(context, programGroupManagerShadow, pipelineManagerShadow, launchParamsManagerShadow,
              "build/shadowRaygen.ptx", "__raygen__shadow", "build/shadowMiss.ptx", "__miss__shadow", "", "",
              "build/shadowAnyHit.ptx", "__anyhit__shadow");

    OptixSBTManager sbtManagerShadow;
    sbtManagerShadow.create(helper.meshGeometriesGPU, programGroupManagerShadow);

    // =========================
    // Create GAS for shared mesh geometries
    // =========================
    std::vector<OptixGAS> gasList;

    // Create GAS for each unique mesh geometry
    for (auto &geometry : helper.meshGeometriesGPU)
    {
        OptixGAS gas;
        // Build GAS from the geometry's vertices and triangles
        gas.build(context, geometry);
        gasList.push_back(std::move(gas));
    }

    // =========================
    // Create instances from GAS with transformations
    // =========================
    std::vector<OptixInstance> instances;
    // Create instances for mesh instances with their transformations
    for (uint32_t i = 0; i < helper.meshInstancesGPU.size(); ++i)
    {
        const MeshInstance &meshInst = helper.meshInstancesGPU[i];

        OptixInstance instance{};

        // Create transformation matrix from quaternion, scale, and translation
        computeTransform(meshInst, instance);

        instance.instanceId = i;
        instance.sbtOffset = meshInst.geometryIndex; // Offset in SBT for this instance

        instance.visibilityMask = 255;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;

        instance.traversableHandle = gasList[meshInst.geometryIndex].handle;

        instances.push_back(instance);
    }

    OptixIAS ias;
    ias.build(context.deviceContext, instances);

    launchParamsManagerRadiance.params.traversable = ias.handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerRadiance.d_params),
                          &launchParamsManagerRadiance.params, sizeof(LaunchRadianceParams), cudaMemcpyHostToDevice));
    launchParamsManagerShadow.params.traversable = ias.handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerShadow.d_params),
                          &launchParamsManagerShadow.params, sizeof(LaunchShadowParams), cudaMemcpyHostToDevice));

    gpuScene.uploadObjects(helper);
    gpuScene.uploadLights(helper);
    gpuScene.uploadMaterials(helper);

    // Now set mesh instances in launch params after uploadObjects has allocated them
    if (gpuScene.meshInstances && gpuScene.nbMeshInstances > 0)
    {
        launchParamsManagerRadiance.params.meshInstances = gpuScene.meshInstances;
        launchParamsManagerRadiance.params.nbMeshInstances = gpuScene.nbMeshInstances;
        // Update launch params on GPU with mesh instance pointers
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerRadiance.d_params),
                              &launchParamsManagerRadiance.params, sizeof(LaunchRadianceParams),
                              cudaMemcpyHostToDevice));

        launchParamsManagerShadow.params.meshInstances = gpuScene.meshInstances;
        launchParamsManagerShadow.params.nbMeshInstances = gpuScene.nbMeshInstances;
        launchParamsManagerShadow.params.materials = gpuScene.materials;
        launchParamsManagerShadow.params.nbMaterials = gpuScene.nbMaterials;
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerShadow.d_params),
                              &launchParamsManagerShadow.params, sizeof(LaunchShadowParams), cudaMemcpyHostToDevice));
    }

    // Radiance
    gpuScene.radiancePass.pipeline = pipelineManagerRadiance.pipeline;
    gpuScene.radiancePass.sbt = sbtManagerRadiance.sbt;

    gpuScene.radiancePass.params = launchParamsManagerRadiance.params;
    gpuScene.radiancePass.d_params = launchParamsManagerRadiance.d_params;

    // Shadow
    gpuScene.shadowPass.pipeline = pipelineManagerShadow.pipeline;
    gpuScene.shadowPass.sbt = sbtManagerShadow.sbt;

    gpuScene.shadowPass.params = launchParamsManagerShadow.params;
    gpuScene.shadowPass.d_params = launchParamsManagerShadow.d_params;
    // Transfer ownership of radiance resources
    pipelineManagerRadiance.pipeline = nullptr;
    sbtManagerRadiance.sbt = {};

    // Transfer ownership of shadow resources
    pipelineManagerShadow.pipeline = nullptr;
    sbtManagerShadow.sbt = {};

    programGroupManagerRadiance.destroy();
    programGroupManagerShadow.destroy();
    return gpuScene;
}
