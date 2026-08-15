#include "../utils/optix_pass_data.cuh"
#include "scene.cuh"

HOST void Scene::uploadObjects(SceneHelper &helper)
{
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
}

HOST void Scene::uploadLights(SceneHelper &helper)
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

HOST void Scene::uploadMaterials(SceneHelper &helper)
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

void sortMaterials(SceneHelper &helper)
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
}

Scene showcase(float3 sunDir, OptixPassData<LaunchRadianceParams> &radiance_pass,
               OptixPassData<LaunchShadowParams> &shadow_pass, float &global_size)
{
    Scene scene;
    SceneHelper helper;
    Light sun = createSun(sunDir, helper);
    helper.lightsGPU.push_back(sun);

    // create scene here

    MeshGeometry bunnyGeometry = loadMeshGeometry("data/bunny/Bunny.obj");
    helper.meshGeometriesGPU.push_back(bunnyGeometry);
    MeshGeometry dragonGeometry = loadMeshGeometry("data/dragon/dragon.obj");
    helper.meshGeometriesGPU.push_back(dragonGeometry);

    // bunny + emissive sphere behind
    Material transparent = Material::makeMaterial(make_float3(1.f), TRANSPARENT, 0.f, 0.f, 1.5f);
    helper.materialsGPU.push_back(transparent);

    Quaternion rotation = quaternionFromAxisAngle(make_float3(0.f, 1.f, 0.f), 180.f);
    float3 scale = make_float3(1.f);
    float3 translation = make_float3(0.f, 1.5f, 0.f);

    MeshInstance instance = createMeshInstance(helper, 0, helper.materialsGPU.size() - 1, scale, rotation, translation);
    helper.meshInstancesGPU.push_back(instance);

    Material emissive = Material::makeMaterial(make_float3(1.f, 0.f, 0.f), EMISSIVE, 0.f, 0.f, 1.f, 11.f);
    helper.materialsGPU.push_back(emissive);
    SDF sdf = SDF::createRandomSphereAnalytic(helper.materialsGPU.size() - 1, 0.5f);
    sdf.translation = make_float3(0.f, 1.5f, -2.f);
    Light light;
    light.metadata = Light::packMetadata(LightType::SDF_GEOM, (int)helper.sdfsGPU.size());
    helper.lightsGPU.push_back(light);
    sdf.lightIndex = (int)helper.lightsGPU.size() - 1;
    helper.sdfsGPU.push_back(sdf);

    // metallic dragon
    Material metal = Material::makeMaterial(make_float3(11.f, 217.f, 121.f)/255.f, METAL, 0.2f * 0.2f, 1.f);
    helper.materialsGPU.push_back(metal);
    Quaternion rotation2 = quaternionFromAxisAngle(make_float3(0.f, 1.f, 0.f), 20.f);
    float3 scale2 = make_float3(15.f);
    float3 translation2 = make_float3(4.f, 0.f, 0.f);
    MeshInstance instance2 = createMeshInstance(helper, 1, helper.materialsGPU.size() - 1, scale2, rotation2, translation2);
    helper.meshInstancesGPU.push_back(instance2);
    // add ground plane last to avoid messing up the mesh instance indices
    addGround(helper);
    // end of scene

    sortLights(helper);
    scene.uploadLights(helper);
    scene.uploadMaterials(helper);
    std::vector<OptixAabb> aabbs;
    for (const auto &sdf : helper.sdfsGPU)
    {
        // aabbs.push_back(sdf.getWorldAABB(primitives, nodes));
        aabbs.push_back(sdf.getWorldAABB());
    }
    global_size += aabbs.size() * sizeof(OptixAabb);
    global_size += helper.sdfsGPU.size() * sizeof(SDF);
    global_size += helper.materialsGPU.size() * sizeof(Material);
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&scene.sdfGeometries.d_aabbBuffer), aabbs.size() * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(scene.sdfGeometries.d_aabbBuffer), aabbs.data(),
                          aabbs.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&scene.sdfGeometries.sdfs), helper.sdfsGPU.size() * sizeof(SDF)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(scene.sdfGeometries.sdfs), helper.sdfsGPU.data(),
                          helper.sdfsGPU.size() * sizeof(SDF), cudaMemcpyHostToDevice));
    scene.sdfGeometries.sdfCount = helper.sdfsGPU.size();

    OptixContext context;
    context.initialize();
    OptixLaunchParamsManager<LaunchRadianceParams> launchParamsManagerRadiance;
    OptixLaunchParamsManager<LaunchShadowParams> launchParamsManagerShadow;
    OptixPipelineManager pipelineManagerRadiance;
    OptixPipelineManager pipelineManagerShadow;
    OptixSBTManager sbtManagerRadiance;
    OptixSBTManager sbtManagerShadow;
    initPassParam(context, launchParamsManagerRadiance, launchParamsManagerShadow, pipelineManagerRadiance,
                  pipelineManagerShadow, sbtManagerRadiance, sbtManagerShadow, scene, helper);

    std::vector<OptixGAS> gasList;

    // Create GAS for each unique mesh geometry
    for (auto &geometry : helper.meshGeometriesGPU)
    {
        OptixGAS gas;
        // Build GAS from the geometry's vertices and triangles
        gas.build(context, geometry, global_size);
        gasList.push_back(std::move(gas));
    }
    if (helper.sdfsGPU.size() > 0)
    {
        OptixGAS sdfGAS;
        sdfGAS.build(context, scene.sdfGeometries, global_size);
        gasList.push_back(std::move(sdfGAS));
    }

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
    if (helper.sdfsGPU.size() > 0)
    {
        OptixInstance sdfInstance{};
        float transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        memcpy(sdfInstance.transform, transform, sizeof(transform));
        sdfInstance.instanceId = helper.meshInstancesGPU.size(); // Next instance ID
        sdfInstance.sbtOffset = helper.meshGeometriesGPU.size(); // Last SBT record
        sdfInstance.visibilityMask = 255;
        sdfInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
        sdfInstance.traversableHandle = gasList.back().handle; // Last GAS is for SDFs
        instances.push_back(sdfInstance);
    }

    OptixIAS ias;
    ias.build(context.deviceContext, instances);

    launchParamsManagerRadiance.params.traversable = ias.handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerRadiance.d_params),
                          &launchParamsManagerRadiance.params, sizeof(LaunchRadianceParams), cudaMemcpyHostToDevice));
    launchParamsManagerShadow.params.traversable = ias.handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerShadow.d_params),
                          &launchParamsManagerShadow.params, sizeof(LaunchShadowParams), cudaMemcpyHostToDevice));

    scene.uploadObjects(helper);

    // Now set mesh instances in launch params after uploadObjects has allocated them
    fillParams(radiance_pass, shadow_pass, pipelineManagerRadiance, pipelineManagerShadow, sbtManagerRadiance,
               sbtManagerShadow, launchParamsManagerRadiance, launchParamsManagerShadow, scene);
    return scene;
}

Scene spheres(float3 sunDir, OptixPassData<LaunchRadianceParams> &radiance_pass,
              OptixPassData<LaunchShadowParams> &shadow_pass, float &global_size, int rngmanip)
{
    Scene scene;
    SceneHelper helper;
    Light sun = createSun(sunDir, helper);
    helper.lightsGPU.push_back(sun);
    for (int i = 0; i < rngmanip; i++)
    {
        float manipRNG = randomFloat();
    }

    addGround(helper);

    int mirrorIndex = helper.materialsGPU.size() - 1;
    float bigRadius = 1.0f;
    float smallRadius = 0.2f;
    float margin = 0.05f;
    float minDist = bigRadius + smallRadius + margin;

    int numberOfSpheresPerSide = 50;
    // ===== PETITES SPHERES =====
    for (int i = -numberOfSpheresPerSide; i < numberOfSpheresPerSide; i++)
    {
        for (int j = -numberOfSpheresPerSide; j < numberOfSpheresPerSide; j++)
        {
            double choose_mat = randomDouble();

            float3 center = make_float3(i + 0.8f * randomFloat(), 0.2f, j + 0.8f * randomFloat());

            if (length(center - make_float3(2.f, 1.f, 1.f)) < minDist ||
                length(center - make_float3(0.f, 1.f, -1.f)) < minDist ||
                length(center - make_float3(-2.f, 1.f, -5.f)) < minDist)
                continue;

            if (choose_mat < 0.40)
            {
                // Lambert coloré
                Material mat = Material::randomLambert();
                helper.materialsGPU.push_back(mat);
            }
            else if (choose_mat < 0.62)
            {
                // Métal coloré
                Material mat = Material::randomMetal();

                helper.materialsGPU.push_back(mat);
            }
            else if (choose_mat < 0.82)
            {
                // Plastique coloré
                Material mat = Material::randomPlastic();

                helper.materialsGPU.push_back(mat);
            }
            else if (choose_mat < 0.90)
            {
                Material mat = Material::makeMaterial(make_float3(1.f), MIRROR);
                helper.materialsGPU.push_back(mat);
            }
            else if (choose_mat < 0.985)
            {
                // Verre coloré aléatoire
                Material mat = Material::randomTransparent();

                helper.materialsGPU.push_back(mat);
            }
            else
            {
                Material mat = Material::randomEmissive();

                helper.materialsGPU.push_back(mat);
            }
            SDF sdf = SDF::createRandomSphereAnalytic(helper.materialsGPU.size() - 1, smallRadius);
            sdf.translation = center;

            // ===== MATERIAL =====
            if (helper.materialsGPU[helper.materialsGPU.size() - 1].type() == EMISSIVE)
            {
                Light light;
                light.metadata = Light::packMetadata(LightType::SDF_GEOM, (int)helper.sdfsGPU.size());
                helper.lightsGPU.push_back(light);
                sdf.lightIndex = (int)helper.lightsGPU.size() - 1;
            }

            helper.sdfsGPU.push_back(sdf);
        }
    }

    Material mirror = Material::makeMaterial(make_float3(1.f, 1.f, 1.f), MIRROR);
    Material transparent = Material::makeMaterial(make_float3(0.9f, 0.9f, 0.9f), TRANSPARENT, 0.f, 0.f, 1.5f);
    Material emissive = Material::makeMaterial(make_float3(1.f, 0.f, 0.f), EMISSIVE, 0.f, 0.f, 1.f, 11.f);
    int mirrorIndex2 = helper.materialsGPU.size();
    helper.materialsGPU.push_back(mirror);
    int transparentIndex = helper.materialsGPU.size();
    helper.materialsGPU.push_back(transparent);
    int emissiveIndex = helper.materialsGPU.size();
    helper.materialsGPU.push_back(emissive);

    SDF sdf = SDF::createRandomSphereAnalytic(mirrorIndex2, bigRadius);
    sdf.translation = make_float3(2.f, 1.f, 1.f);
    helper.sdfsGPU.push_back(sdf);
    sdf = SDF::createRandomSphereAnalytic(transparentIndex, bigRadius);
    sdf.translation = make_float3(0.f, 1.f, -1.f);
    helper.sdfsGPU.push_back(sdf);
    sdf = SDF::createRandomSphereAnalytic(emissiveIndex, bigRadius);
    sdf.translation = make_float3(-2.f, 1.f, -5.f);
    Light light;
    light.metadata = Light::packMetadata(LightType::SDF_GEOM, (int)helper.sdfsGPU.size());
    helper.lightsGPU.push_back(light);
    sdf.lightIndex = (int)helper.lightsGPU.size() - 1;
    helper.sdfsGPU.push_back(sdf);

    sortLights(helper);
    scene.uploadLights(helper);
    scene.uploadMaterials(helper);
    std::vector<OptixAabb> aabbs;
    for (const auto &sdf : helper.sdfsGPU)
    {
        // aabbs.push_back(sdf.getWorldAABB(primitives, nodes));
        aabbs.push_back(sdf.getWorldAABB());
    }
    global_size += aabbs.size() * sizeof(OptixAabb);
    global_size += helper.sdfsGPU.size() * sizeof(SDF);
    global_size += helper.materialsGPU.size() * sizeof(Material);
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&scene.sdfGeometries.d_aabbBuffer), aabbs.size() * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(scene.sdfGeometries.d_aabbBuffer), aabbs.data(),
                          aabbs.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&scene.sdfGeometries.sdfs), helper.sdfsGPU.size() * sizeof(SDF)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(scene.sdfGeometries.sdfs), helper.sdfsGPU.data(),
                          helper.sdfsGPU.size() * sizeof(SDF), cudaMemcpyHostToDevice));
    scene.sdfGeometries.sdfCount = helper.sdfsGPU.size();

    OptixContext context;
    context.initialize();
    OptixLaunchParamsManager<LaunchRadianceParams> launchParamsManagerRadiance;
    OptixLaunchParamsManager<LaunchShadowParams> launchParamsManagerShadow;
    OptixPipelineManager pipelineManagerRadiance;
    OptixPipelineManager pipelineManagerShadow;
    OptixSBTManager sbtManagerRadiance;
    OptixSBTManager sbtManagerShadow;
    initPassParam(context, launchParamsManagerRadiance, launchParamsManagerShadow, pipelineManagerRadiance,
                  pipelineManagerShadow, sbtManagerRadiance, sbtManagerShadow, scene, helper);

    std::vector<OptixGAS> gasList;

    // Create GAS for each unique mesh geometry
    for (auto &geometry : helper.meshGeometriesGPU)
    {
        OptixGAS gas;
        // Build GAS from the geometry's vertices and triangles
        gas.build(context, geometry, global_size);
        gasList.push_back(std::move(gas));
    }
    if (helper.sdfsGPU.size() > 0)
    {
        OptixGAS sdfGAS;
        sdfGAS.build(context, scene.sdfGeometries, global_size);
        gasList.push_back(std::move(sdfGAS));
    }

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
    if (helper.sdfsGPU.size() > 0)
    {
        OptixInstance sdfInstance{};
        float transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        memcpy(sdfInstance.transform, transform, sizeof(transform));
        sdfInstance.instanceId = helper.meshInstancesGPU.size(); // Next instance ID
        sdfInstance.sbtOffset = helper.meshGeometriesGPU.size(); // Last SBT record
        sdfInstance.visibilityMask = 255;
        sdfInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
        sdfInstance.traversableHandle = gasList.back().handle; // Last GAS is for SDFs
        instances.push_back(sdfInstance);
    }

    OptixIAS ias;
    ias.build(context.deviceContext, instances);

    launchParamsManagerRadiance.params.traversable = ias.handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerRadiance.d_params),
                          &launchParamsManagerRadiance.params, sizeof(LaunchRadianceParams), cudaMemcpyHostToDevice));
    launchParamsManagerShadow.params.traversable = ias.handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerShadow.d_params),
                          &launchParamsManagerShadow.params, sizeof(LaunchShadowParams), cudaMemcpyHostToDevice));

    scene.uploadObjects(helper);

    // Now set mesh instances in launch params after uploadObjects has allocated them
    fillParams(radiance_pass, shadow_pass, pipelineManagerRadiance, pipelineManagerShadow, sbtManagerRadiance,
               sbtManagerShadow, launchParamsManagerRadiance, launchParamsManagerShadow, scene);

    return scene;
}

Scene loadScene(float3 sunDir, OptixPassData<LaunchRadianceParams> &radiance_pass,
                OptixPassData<LaunchShadowParams> &shadow_pass, float &global_size, int rngmanip)
{
    Scene scene;
    SceneHelper helper;
    Light sun = createSun(sunDir, helper);
    helper.lightsGPU.push_back(sun);

    // changes random scene
    for (int i = 0; i < rngmanip; i++)
    {
        float manipRNG = randomFloat();
    }
    // create materials
    createMaterials(helper);

    // ===== MESH INSTANCING: Load geometry once, create multiple instances =====
    MeshGeometry bunnyGeometry = loadMeshGeometry("data/bunny/Bunny.obj");
    helper.meshGeometriesGPU.push_back(bunnyGeometry);
    MeshGeometry dragonGeometry = loadMeshGeometry("data/dragon/dragon.obj", make_float3(10.f));
    helper.meshGeometriesGPU.push_back(dragonGeometry);

    // Create instances with different transforms and materials
    for (int i = 0; i < 0; i++)
    {
        Quaternion rotation = quaternionFromAxisAngle(
            make_float3(randomFloat() * 2.f, randomFloat() * 2.f, randomFloat() * 2.f), randomFloat() * 360.f);

        float3 scale = make_float3(randomFloat() * 0.5f + 0.5f);
        float3 translation =
            make_float3(randomFloat() * 10.f - 4.f, randomFloat() * 10.f - 6.f, -randomFloat() * 10.f + 4.f);

        int materialIndex = int(randomFloat() * helper.materialsGPU.size());

        // Create an instance (no GPU allocation here, just structure setup)
        MeshInstance instance = createMeshInstance(helper, int(randomFloat() * helper.meshGeometriesGPU.size()),
                                                   materialIndex, scale, rotation, translation);
        if (helper.materialsGPU[materialIndex].type() == EMISSIVE)
        {
            Light light;
            light.metadata = Light::packMetadata(LightType::MESH_GEOM, (int)helper.meshInstancesGPU.size());
            helper.lightsGPU.push_back(light);
            instance.lightIndex = (int)helper.lightsGPU.size() - 1;
        }
        helper.meshInstancesGPU.push_back(instance);
    }

    // Add ground plane
    addGround(helper);

    // Add spheres SDFs
    for (int i = 0; i < 0; i++)
    {
        int materialIndex = int(randomFloat() * helper.materialsGPU.size());
        SDF sdf = SDF::createRandomSphereAnalytic(materialIndex);
        if (helper.materialsGPU[materialIndex].type() == EMISSIVE)
        {
            Light light;
            light.metadata = Light::packMetadata(LightType::SDF_GEOM, (int)helper.sdfsGPU.size());
            helper.lightsGPU.push_back(light);
            sdf.lightIndex = (int)helper.lightsGPU.size() - 1;
        }
        helper.sdfsGPU.push_back(sdf);
    }
    SDF sdf = load_sdf("data/bunnySDF/bunny_256.sdf");
    sdf.translation = make_float3(0.f, 2.f, 0.f);
    sdf.signedGrid.materialIndex = 1;
    sdf.aabb = sdf.getAABB();
    helper.sdfsGPU.push_back(sdf);
    /*
    // Add torus SDFs
    for (int i = 0; i < 15; i++)
    {
        int materialIndex = int(randomFloat() * helper.materialsGPU.size());
        SDF sdf = SDF::createRandomToreSDF(materialIndex);
        if (helper.materialsGPU[materialIndex].type() == EMISSIVE)
        {
            Light light;
            light.metadata = Light::packMetadata(LightType::SDF_GEOM, (int)helper.sdfsGPU.size());
            helper.lightsGPU.push_back(light);
            sdf.lightIndex = (int)helper.lightsGPU.size() - 1;
        }
        helper.sdfsGPU.push_back(sdf);
    }*/

    // Add CSG SDFs
    /*helper.materialsGPU.push_back(Material::makeMaterial(make_float3(1.f), TRANSPARENT, 0.f, 0.f, 1.5f));
    int materialIndex = helper.materialsGPU.size() - 1;
    SDF sdf;
    sdf.type = SDFType::CSGTree;
    CSGTree csgTree;
    csgTree.materialIndex = materialIndex;
    std::vector<PrimitiveData> primitives;
    for(int i = 0; i < 10; i++)
    {
        primitives.push_back(PrimitiveData::createSpherePrimitive(make_float3(randomFloat() * 10.f - 5.f, randomFloat()
    * 10.f - 5.f, randomFloat() * 10.f - 5.f), materialIndex, randomFloat() * 0.5f + 0.2f));
    }
    /*Quaternion rotation = quaternionFromAxisAngle(
        make_float3(0.f, 0.f, 1.f), 180.f);
        Matrix3x3 rotationMatrix = quaternionToMatrix(rotation);
    primitives.push_back(PrimitiveData::createConePrimitive(rotationMatrix, make_float3(0.f, 0.f, 0.5f),
    materialIndex, 1.6f, 35.f));

    std::vector<CSGNode> nodes;
    csgTree.buildTree(0, primitives.size() - 1, nodes);
    csgTree.compile(nodes);

    // Allocate and upload primitives to GPU
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&csgTree.primArray), primitives.size() * sizeof(PrimitiveData)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(csgTree.primArray), primitives.data(),
                          primitives.size() * sizeof(PrimitiveData), cudaMemcpyHostToDevice));

    sdf.csgTree = csgTree;
    sdf.translation = make_float3(0.f, 0.f, 0.f);
    sdf.aabb = sdf.getAABB(primitives, nodes);
    helper.sdfsGPU.push_back(sdf);*/

    // Sort lights by power for importance sampling
    // Upload now because of emissive materials that are added to the lights list
    sortLights(helper);
    scene.uploadLights(helper);
    scene.uploadMaterials(helper);

    if (helper.sdfsGPU.size() > 0)
    {

        std::vector<OptixAabb> aabbs;
        for (const auto &sdf : helper.sdfsGPU)
        {
            // aabbs.push_back(sdf.getWorldAABB(primitives, nodes));
            aabbs.push_back(sdf.getWorldAABB());
        }
        global_size += aabbs.size() * sizeof(OptixAabb);
        global_size += helper.sdfsGPU.size() * sizeof(SDF);
        global_size += helper.materialsGPU.size() * sizeof(Material);
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&scene.sdfGeometries.d_aabbBuffer), aabbs.size() * sizeof(OptixAabb)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(scene.sdfGeometries.d_aabbBuffer), aabbs.data(),
                              aabbs.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&scene.sdfGeometries.sdfs), helper.sdfsGPU.size() * sizeof(SDF)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(scene.sdfGeometries.sdfs), helper.sdfsGPU.data(),
                              helper.sdfsGPU.size() * sizeof(SDF), cudaMemcpyHostToDevice));
        scene.sdfGeometries.sdfCount = helper.sdfsGPU.size();
    }

    OptixContext context;
    context.initialize();
    OptixLaunchParamsManager<LaunchRadianceParams> launchParamsManagerRadiance;
    OptixLaunchParamsManager<LaunchShadowParams> launchParamsManagerShadow;
    OptixPipelineManager pipelineManagerRadiance;
    OptixPipelineManager pipelineManagerShadow;
    OptixSBTManager sbtManagerRadiance;
    OptixSBTManager sbtManagerShadow;
    initPassParam(context, launchParamsManagerRadiance, launchParamsManagerShadow, pipelineManagerRadiance,
                  pipelineManagerShadow, sbtManagerRadiance, sbtManagerShadow, scene, helper);

    // =========================
    // Create GAS for shared mesh geometries
    // =========================
    std::vector<OptixGAS> gasList;

    // Create GAS for each unique mesh geometry
    for (auto &geometry : helper.meshGeometriesGPU)
    {
        OptixGAS gas;
        // Build GAS from the geometry's vertices and triangles
        gas.build(context, geometry, global_size);
        gasList.push_back(std::move(gas));
    }
    if (helper.sdfsGPU.size() > 0)
    {
        OptixGAS sdfGAS;
        sdfGAS.build(context, scene.sdfGeometries, global_size);
        gasList.push_back(std::move(sdfGAS));
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
    if (helper.sdfsGPU.size() > 0)
    {
        OptixInstance sdfInstance{};
        float transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        memcpy(sdfInstance.transform, transform, sizeof(transform));
        sdfInstance.instanceId = helper.meshInstancesGPU.size(); // Next instance ID
        sdfInstance.sbtOffset = helper.meshGeometriesGPU.size(); // Last SBT record
        sdfInstance.visibilityMask = 255;
        sdfInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
        sdfInstance.traversableHandle = gasList.back().handle; // Last GAS is for SDFs
        instances.push_back(sdfInstance);
    }

    OptixIAS ias;
    ias.build(context.deviceContext, instances);

    launchParamsManagerRadiance.params.traversable = ias.handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerRadiance.d_params),
                          &launchParamsManagerRadiance.params, sizeof(LaunchRadianceParams), cudaMemcpyHostToDevice));
    launchParamsManagerShadow.params.traversable = ias.handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerShadow.d_params),
                          &launchParamsManagerShadow.params, sizeof(LaunchShadowParams), cudaMemcpyHostToDevice));

    scene.uploadObjects(helper);

    // Now set mesh instances in launch params after uploadObjects has allocated them

    fillParams(radiance_pass, shadow_pass, pipelineManagerRadiance, pipelineManagerShadow, sbtManagerRadiance,
               sbtManagerShadow, launchParamsManagerRadiance, launchParamsManagerShadow, scene);

    return scene;
}

void addGround(SceneHelper &helper)
{
    Plane p = Plane(make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f));
    Material ground = Material::makeMaterial(make_float3(0.5f), LAMBERT, 1.0f);
    helper.materialsGPU.push_back(ground);
    p.materialIndex = helper.materialsGPU.size() - 1;
    helper.meshGeometriesGPU.push_back(PlaneToMesh(p, 20000.f));
    helper.meshInstancesGPU.push_back(createMeshInstance(helper, helper.meshGeometriesGPU.size() - 1, p.materialIndex));
}

void initPassParam(OptixContext &context, OptixLaunchParamsManager<LaunchRadianceParams> &launchParamsManagerRadiance,
                   OptixLaunchParamsManager<LaunchShadowParams> &launchParamsManagerShadow,
                   OptixPipelineManager &pipelineManagerRadiance, OptixPipelineManager &pipelineManagerShadow,
                   OptixSBTManager &sbtManagerRadiance, OptixSBTManager &sbtManagerShadow, Scene &scene,
                   SceneHelper &helper)
{
    OptixProgramGroupManager programGroupManagerRadiance;
    programGroupManagerRadiance.addRaygenProgram(context, "build/radiance_raygen.ptx", "__raygen__radiance");
    programGroupManagerRadiance.addMissProgram(context, "build/radiance_miss.ptx", "__miss__radiance");
    programGroupManagerRadiance.addMeshHitProgram(context, "build/radiance_closest_hit.ptx", "__closesthit__radiance",
                                                  "", "", "", "");
    programGroupManagerRadiance.addSdfHitProgram(context, "build/radiance_sdf_closest_hit.ptx",
                                                 "__closesthit__radiance__sdf", "", "",
                                                 "build/radiance_sdf_intersection.ptx", "__intersection__sdf");

    initOptix(context, programGroupManagerRadiance, pipelineManagerRadiance, launchParamsManagerRadiance);

    sbtManagerRadiance.create(helper.meshGeometriesGPU, programGroupManagerRadiance, scene.sdfGeometries);

    OptixProgramGroupManager programGroupManagerShadow;
    programGroupManagerShadow.addRaygenProgram(context, "build/shadow_raygen.ptx", "__raygen__shadow");
    programGroupManagerShadow.addMissProgram(context, "build/shadow_miss.ptx", "__miss__shadow");
    programGroupManagerShadow.addMeshHitProgram(context, "", "", "build/shadow_any_hit.ptx", "__anyhit__shadow", "",
                                                "");
    programGroupManagerShadow.addSdfHitProgram(context, "", "", "build/shadow_sdf_any_hit.ptx", "__anyhit__shadow__sdf",
                                               "build/shadow_sdf_intersection.ptx", "__intersection__sdf__shadow");
    initOptix(context, programGroupManagerShadow, pipelineManagerShadow, launchParamsManagerShadow);

    sbtManagerShadow.create(helper.meshGeometriesGPU, programGroupManagerShadow, scene.sdfGeometries);
    programGroupManagerRadiance.destroy();
    programGroupManagerShadow.destroy();
}

void fillParams(OptixPassData<LaunchRadianceParams> &radiance_pass, OptixPassData<LaunchShadowParams> &shadow_pass,
                OptixPipelineManager &pipelineManagerRadiance, OptixPipelineManager &pipelineManagerShadow,
                OptixSBTManager &sbtManagerRadiance, OptixSBTManager &sbtManagerShadow,
                OptixLaunchParamsManager<LaunchRadianceParams> &launchParamsManagerRadiance,
                OptixLaunchParamsManager<LaunchShadowParams> &launchParamsManagerShadow, Scene &scene)
{

    launchParamsManagerRadiance.params.meshInstances = scene.meshInstances;
    launchParamsManagerRadiance.params.nbMeshInstances = scene.nbMeshInstances;
    launchParamsManagerRadiance.params.materials = scene.materials;
    launchParamsManagerRadiance.params.nbMaterials = scene.nbMaterials;
    // Update launch params on GPU with mesh instance pointers
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerRadiance.d_params),
                          &launchParamsManagerRadiance.params, sizeof(LaunchRadianceParams), cudaMemcpyHostToDevice));

    launchParamsManagerShadow.params.meshInstances = scene.meshInstances;
    launchParamsManagerShadow.params.nbMeshInstances = scene.nbMeshInstances;
    launchParamsManagerShadow.params.materials = scene.materials;
    launchParamsManagerShadow.params.nbMaterials = scene.nbMaterials;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerShadow.d_params),
                          &launchParamsManagerShadow.params, sizeof(LaunchShadowParams), cudaMemcpyHostToDevice));

    radiance_pass.pipeline = pipelineManagerRadiance.pipeline;
    radiance_pass.sbt = sbtManagerRadiance.sbt;

    radiance_pass.params = launchParamsManagerRadiance.params;
    radiance_pass.d_params = launchParamsManagerRadiance.d_params;

    // Shadow
    shadow_pass.pipeline = pipelineManagerShadow.pipeline;
    shadow_pass.sbt = sbtManagerShadow.sbt;

    shadow_pass.params = launchParamsManagerShadow.params;
    shadow_pass.d_params = launchParamsManagerShadow.d_params;
    // Transfer ownership of radiance resources
    pipelineManagerRadiance.pipeline = nullptr;
    sbtManagerRadiance.sbt = {};

    // Transfer ownership of shadow resources
    pipelineManagerShadow.pipeline = nullptr;
    sbtManagerShadow.sbt = {};
}

void sortLights(SceneHelper &helper)
{
    std::sort(helper.lightsGPU.begin(), helper.lightsGPU.end(), [](const Light &a, const Light &b) {
        return a.getColorPower().x + a.getColorPower().y + a.getColorPower().z >
               b.getColorPower().x + b.getColorPower().y + b.getColorPower().z;
    });
}

void createMaterials(SceneHelper &helper, int rngmanip)
{
    for (int i = 0; i < 5; i++)
    {
        Material emissive = Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()), EMISSIVE,
                                                   0.f, 0.f, 1.f, randomFloat() * 5.f + 8.f);
        helper.materialsGPU.push_back(emissive);
    }
    for (int i = 0; i < 10; i++) // 5-14
    {
        Material mirror = Material::makeMaterial(make_float3(1.f), MIRROR);
        helper.materialsGPU.push_back(mirror);
    }

    for (int i = 0; i < 15; i++) // 15 -29
    {
        Material transparent = Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()),
                                                      TRANSPARENT, 0.f, 0.f, 1.5f);
        helper.materialsGPU.push_back(transparent);
    }
    for (int i = 0; i < 10; i++) // 30 - 39
    {
        Material lambert =
            Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()), LAMBERT, 1.0f);
        helper.materialsGPU.push_back(lambert);
    }

    for (int i = 0; i < 10; i++) // 40 - 49
    {
        float roughness = randomFloat() * 0.5f;
        Material metal = Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()), METAL,
                                                roughness * roughness, 1.f);
        helper.materialsGPU.push_back(metal);
    }

    for (int i = 0; i < 10; i++) // 50 - 59
    {
        float roughness = randomFloat() * 0.5f;
        Material plastic = Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()), PLASTIC,
                                                  roughness * roughness);
        helper.materialsGPU.push_back(plastic);
    }
}