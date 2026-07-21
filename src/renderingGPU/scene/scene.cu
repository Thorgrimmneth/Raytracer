#include "scene.cuh"
#include "../utils/optix_pass_data.cuh"

HOST void Scene::uploadObjects(SceneHelper &helper)
{
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

Scene loadScene(float3 sunDir, OptixPassData<LaunchRadianceParams> &radiance_pass, OptixPassData<LaunchShadowParams> &shadow_pass, int rngmanip)
{
    Scene gpuScene;
    SceneHelper helper;
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

    for (int i = 0; i < 15; i++)
    {
        Material transparent = Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()),
                                                      TRANSPARENT, 0.f, 0.f, 1.5f);
        helper.materialsGPU.push_back(transparent);
    }
    for (int i = 0; i < 10; i++)
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

    for (int i = 0; i < 15; i++)
    {
        int materialIndex = int(randomFloat() * helper.materialsGPU.size());
        SDF sdf = SDF::createRandomSphereSDF(materialIndex);
        sdf.translation =
            make_float3(randomFloat() * 10.f - 5.f, randomFloat() * 10.f - 5.f, randomFloat() * 10.f - 5.f);
        if (helper.materialsGPU[materialIndex].type() == EMISSIVE)
        {
            Light light;
            light.metadata = Light::packMetadata(LightType::SDF_GEOM, (int)helper.sdfsGPU.size());
            helper.lightsGPU.push_back(light);
            sdf.lightIndex = (int)helper.lightsGPU.size() - 1;
        }
        helper.sdfsGPU.push_back(sdf);
    }

    for (int i = 0; i < 15; i++)
    {
        int materialIndex = int(randomFloat() * helper.materialsGPU.size());
        Quaternion rotation = quaternionFromAxisAngle(
            make_float3(randomFloat() * 2.f, randomFloat() * 2.f, randomFloat() * 2.f), randomFloat() * 360.f);
        SDF sdf = SDF::createRandomToreSDF(materialIndex);
        Matrix3x3 rotationMatrix = quaternionToMatrix(rotation);
        sdf.rotation = rotationMatrix.transpose();
        sdf.translation =
            make_float3(randomFloat() * 10.f - 5.f, randomFloat() * 10.f - 5.f, randomFloat() * 10.f - 5.f);
        if (helper.materialsGPU[materialIndex].type() == EMISSIVE)
        {
            Light light;
            light.metadata = Light::packMetadata(LightType::SDF_GEOM, (int)helper.sdfsGPU.size());
            helper.lightsGPU.push_back(light);
            sdf.lightIndex = (int)helper.lightsGPU.size() - 1;
        }
        helper.sdfsGPU.push_back(sdf);
    }

    sortLights(helper);
    gpuScene.uploadLights(helper);
    gpuScene.uploadMaterials(helper);
    std::vector<OptixAabb> aabbs;
    for (const auto &sdf : helper.sdfsGPU)
    {
        aabbs.push_back(sdf.getWorldAABB());
    }

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&gpuScene.sdfGeometries.d_aabbBuffer), aabbs.size() * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(gpuScene.sdfGeometries.d_aabbBuffer), aabbs.data(),
                          aabbs.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&gpuScene.sdfGeometries.sdfs), helper.sdfsGPU.size() * sizeof(SDF)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(gpuScene.sdfGeometries.sdfs), helper.sdfsGPU.data(),
                          helper.sdfsGPU.size() * sizeof(SDF), cudaMemcpyHostToDevice));
    gpuScene.sdfGeometries.sdfCount = helper.sdfsGPU.size();

    OptixContext context;
    context.initialize();
    OptixProgramGroupManager programGroupManagerRadiance;
    OptixPipelineManager pipelineManagerRadiance;
    OptixLaunchParamsManager<LaunchRadianceParams> launchParamsManagerRadiance;
    programGroupManagerRadiance.addRaygenProgram(context, "build/radiance_raygen.ptx", "__raygen__radiance");
    programGroupManagerRadiance.addMissProgram(context, "build/radiance_miss.ptx", "__miss__radiance");
    programGroupManagerRadiance.addMeshHitProgram(context, "build/radiance_closest_hit.ptx", "__closesthit__radiance",
                                                  "", "", "", "");
    programGroupManagerRadiance.addSdfHitProgram(context, "build/radiance_sdf_closest_hit.ptx",
                                                 "__closesthit__radiance__sdf", "", "",
                                                 "build/radiance_sdf_intersection.ptx", "__intersection__sdf");

    initOptix(context, programGroupManagerRadiance, pipelineManagerRadiance, launchParamsManagerRadiance);

    OptixSBTManager sbtManagerRadiance;
    sbtManagerRadiance.create(helper.meshGeometriesGPU, programGroupManagerRadiance, gpuScene.sdfGeometries);

    OptixProgramGroupManager programGroupManagerShadow;
    OptixPipelineManager pipelineManagerShadow;
    OptixLaunchParamsManager<LaunchShadowParams> launchParamsManagerShadow;
    programGroupManagerShadow.addRaygenProgram(context, "build/shadow_raygen.ptx", "__raygen__shadow");
    programGroupManagerShadow.addMissProgram(context, "build/shadow_miss.ptx", "__miss__shadow");
    programGroupManagerShadow.addMeshHitProgram(context, "", "", "build/shadow_any_hit.ptx", "__anyhit__shadow", "",
                                                "");
    programGroupManagerShadow.addSdfHitProgram(context, "", "", "build/shadow_sdf_any_hit.ptx", "__anyhit__shadow__sdf",
                                               "build/shadow_sdf_intersection.ptx", "__intersection__sdf__shadow");
    initOptix(context, programGroupManagerShadow, pipelineManagerShadow, launchParamsManagerShadow);

    OptixSBTManager sbtManagerShadow;
    sbtManagerShadow.create(helper.meshGeometriesGPU, programGroupManagerShadow, gpuScene.sdfGeometries);

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
    OptixGAS sdfGAS;
    sdfGAS.build(context, gpuScene.sdfGeometries);
    gasList.push_back(std::move(sdfGAS));
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

    OptixInstance sdfInstance{};

    float transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

    memcpy(sdfInstance.transform, transform, sizeof(transform));

    sdfInstance.instanceId = helper.meshInstancesGPU.size();

    sdfInstance.sbtOffset = helper.meshGeometriesGPU.size(); // dernier record SBT

    sdfInstance.visibilityMask = 255;
    sdfInstance.flags = OPTIX_INSTANCE_FLAG_NONE;

    sdfInstance.traversableHandle = gasList.back().handle;

    instances.push_back(sdfInstance);

    OptixIAS ias;
    ias.build(context.deviceContext, instances);

    launchParamsManagerRadiance.params.traversable = ias.handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerRadiance.d_params),
                          &launchParamsManagerRadiance.params, sizeof(LaunchRadianceParams), cudaMemcpyHostToDevice));
    launchParamsManagerShadow.params.traversable = ias.handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(launchParamsManagerShadow.d_params),
                          &launchParamsManagerShadow.params, sizeof(LaunchShadowParams), cudaMemcpyHostToDevice));

    gpuScene.uploadObjects(helper);

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

    programGroupManagerRadiance.destroy();
    programGroupManagerShadow.destroy();
    return gpuScene;
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

void sortLights(SceneHelper &helper)
{
    std::sort(helper.lightsGPU.begin(), helper.lightsGPU.end(), [](const Light &a, const Light &b) {
        return a.getColorPower().x + a.getColorPower().y + a.getColorPower().z >
               b.getColorPower().x + b.getColorPower().y + b.getColorPower().z;
    });
}